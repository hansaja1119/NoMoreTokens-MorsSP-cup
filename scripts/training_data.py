"""Scene-uniform crops with deterministic per-sample augmentation."""
from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset
from common import SCRIPTS,load_split
from classical import features

class TrainingCrops(Dataset):
    def __init__(self,steps,batch,hybrid=True,synthetic=True,seed=20260908,start_step=0,warmup_steps=1000):
        self.ids=load_split('train');self.steps=steps;self.batch=batch;self.hybrid=hybrid
        self.synthetic=synthetic;self.seed=seed;self.start_step=start_step;self.warmup_steps=warmup_steps
        self.arrays={}
    def __len__(self):return (self.steps-self.start_step)*self.batch
    def __getitem__(self,index):
        absolute=index+self.start_step*self.batch
        rng=np.random.default_rng(self.seed+absolute)
        image_id=self.ids[int(rng.integers(len(self.ids)))]
        if image_id not in self.arrays:
            self.arrays[image_id]=np.load(SCRIPTS/'cache'/'features_v1'/f'{image_id}.npy',mmap_mode='r')
        arr=self.arrays[image_id]
        size=128 if absolute//self.batch<self.warmup_steps else 256
        top=int(rng.integers(arr.shape[0]-size+1));left=int(rng.integers(arr.shape[1]-size+1))
        crop=np.asarray(arr[top:top+size,left:left+size],dtype=np.float32)/255
        inp=crop[...,:9].copy();inp[...,6:9]*=.3
        target=crop[...,9:12]
        branch=rng.random() if self.synthetic else 0.
        if branch>=.7:
            if branch>=.9:
                noisy=target.copy()
            else:
                # Broad photographic corruption, not reconstruction of organizer RNG.
                read=float(rng.uniform(.003,.07));shot=float(rng.uniform(0,.025))
                sigma=np.sqrt(read*read+shot*np.maximum(target,0))
                sigma*=np.linspace(rng.uniform(.6,1.4),rng.uniform(.6,1.4),size,dtype=np.float32)[None,:,None]
                noisy=target+rng.normal(size=target.shape).astype(np.float32)*sigma
                if rng.random()<.5:
                    mask=rng.random(target.shape)<rng.uniform(0,.008)
                    noisy=np.where(mask,rng.integers(0,2,size=target.shape),noisy)
                noisy=np.rint(np.clip(noisy,0,1)*255).astype(np.float32)/255
            inp=features(noisy) if self.hybrid else noisy
        if not self.hybrid:inp=inp[...,:3]
        rot=int(rng.integers(4));inp=np.rot90(inp,rot);target=np.rot90(target,rot)
        if rng.random()<.5:inp=inp[:,::-1];target=target[:,::-1]
        return torch.from_numpy(np.ascontiguousarray(inp.transpose(2,0,1))),torch.from_numpy(np.ascontiguousarray(target.transpose(2,0,1)))
