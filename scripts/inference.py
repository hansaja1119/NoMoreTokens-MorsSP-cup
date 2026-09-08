"""Deterministic float32 local inference with memory-bounded overlap blending."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from classical import features,noise_maps,OPPONENT
from model import Denoiser

def choose_device(name='auto'):
    if name=='auto':name='cuda' if torch.cuda.is_available() else 'cpu'
    if name=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; use --device cpu')
    return torch.device(name)

def load_model(checkpoint: Path,device):
    if not checkpoint.is_file():raise FileNotFoundError(f'Missing local checkpoint: {checkpoint}. See scripts/README.md.')
    obj=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=Denoiser(**obj['model_config'])
    model.load_state_dict(obj['ema'] if 'ema' in obj else obj['model'])
    model.mild_threshold=obj.get('inference_config',{}).get('mild_threshold',0.)
    return model.to(device).eval(),obj

def protect_mild(rgb,pred,sigma,threshold):
    if threshold<=0:return pred
    # Fine scene detail is often shared by channels. Opponent chroma helps
    # avoid interpreting that common texture as independent sensor noise.
    estimate=noise_maps(rgb@OPPONENT.T)[...,1:].mean(axis=-1,keepdims=True)
    confidence=np.clip(estimate/threshold,0,1)**2
    # Chroma alone would miss correlated or grayscale noise. The low-eigenvalue
    # variance of weak-texture luminance patches provides independent evidence.
    luma_sigma=weak_texture_luma_sigma(rgb)
    luma_confidence=float(np.clip((luma_sigma-.02)/.03,0,1)**2)
    confidence=np.maximum(confidence,luma_confidence)
    return rgb+confidence*(pred-rgb)

def weak_texture_luma_sigma(rgb):
    h,w=rgb.shape[:2]
    if min(h,w)<16:return 0.
    stride=max(4,int(np.sqrt(h*w/4096)))
    patches=np.lib.stride_tricks.sliding_window_view(rgb.mean(-1),(7,7))[::stride,::stride].reshape(-1,49)
    if len(patches)<64:return 0.
    variance=patches.var(1)
    weak=patches[variance<=np.quantile(variance,.25)]
    if len(weak)<16:return 0.
    centered=weak-weak.mean(0,keepdims=True)
    eigenvalues=np.linalg.eigvalsh(centered.T@centered/(len(centered)-1))
    return float(np.sqrt(np.maximum(eigenvalues[:8],0).mean()))

def positions(length,tile,overlap):
    if length<=tile:return [0]
    stride=tile-overlap
    return sorted(set(list(range(0,length-tile+1,stride))+[length-tile]))

@torch.inference_mode()
def predict(model,rgb,device,tile=512,overlap=64,mild_threshold=None):
    if tile<16 or overlap<0 or overlap>=tile:
        raise ValueError('Require tile >=16 and 0 <= overlap < tile')
    z=features(rgb) if model.config['hybrid'] else rgb
    h,w=rgb.shape[:2]
    output=np.zeros((h,w,3),np.float32);weights=np.zeros((h,w,1),np.float32)
    for top in positions(h,tile,overlap):
        for left in positions(w,tile,overlap):
            a=z[top:top+tile,left:left+tile]
            tensor=torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1))).unsqueeze(0).to(device)
            result=model(tensor).squeeze(0).permute(1,2,0).cpu().numpy()
            ah,aw=a.shape[:2]
            # Positive endpoints retain exterior pixels and support odd/tiny images.
            wy=np.hanning(ah+2)[1:-1];wx=np.hanning(aw+2)[1:-1]
            weight=np.maximum(wy[:,None]*wx[None,:],1e-4).astype(np.float32)[...,None]
            output[top:top+ah,left:left+aw]+=result*weight
            weights[top:top+ah,left:left+aw]+=weight
    threshold=getattr(model,'mild_threshold',0.) if mild_threshold is None else mild_threshold
    sigma=z[...,6:9] if model.config['hybrid'] else noise_maps(rgb) if threshold>0 else None
    return protect_mild(rgb,output/weights,sigma,threshold)
