"""Use the unmodified official evaluator as the metric source of truth."""
from __future__ import annotations
import importlib.util
import numpy as np
from common import ROOT

def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod

official=load_module('official_evaluator',ROOT/'evaluation'/'evaluate.py')

def score_arrays(noisy,pred,gt):
    if noisy.shape!=gt.shape or pred.shape!=gt.shape:
        raise ValueError('Image shape mismatch')
    if not all(np.isfinite(a).all() for a in [noisy,pred,gt]):
        raise ValueError('Non-finite image values')
    psnr=float(official.sk_psnr(gt,pred,data_range=1.0))
    ssim=official.ssim_value(gt,pred)
    np_=float(official.sk_psnr(gt,noisy,data_range=1.0))
    ns=official.ssim_value(gt,noisy)
    return dict(psnr=psnr,ssim=ssim,noisy_psnr=np_,noisy_ssim=ns,
                delta_psnr=psnr-np_,delta_ssim=ssim-ns,
                composite_score=official.official_composite(psnr-np_,ssim-ns))

def summarize(rows):
    keys=['psnr','ssim','noisy_psnr','noisy_ssim','delta_psnr','delta_ssim','composite_score','seconds']
    out={k:float(np.mean([r[k] for r in rows])) for k in keys if k in rows[0]}
    out['count']=len(rows)
    out['median_seconds']=float(np.median([r['seconds'] for r in rows]))
    out['worst_decile_composite']=float(np.mean(sorted(r['composite_score'] for r in rows)[:max(1,len(rows)//10)]))
    return out
