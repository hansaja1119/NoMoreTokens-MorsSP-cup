"""Input-adaptive signal processing, independent of filenames and reference images."""
from __future__ import annotations
import cv2
import numpy as np
import pywt

# Orthonormal opponent coordinates preserve white-noise variance.
OPPONENT=np.array([[1,1,1],[1,0,-1],[1,-2,1]],np.float32)
OPPONENT/=np.linalg.norm(OPPONENT,axis=1,keepdims=True)

def noise_maps(rgb: np.ndarray, block: int=64) -> np.ndarray:
    h,w,_=rgb.shape
    grids=np.zeros(((h+block-1)//block,(w+block-1)//block,3),np.float32)
    for gy,top in enumerate(range(0,h,block)):
        for gx,left in enumerate(range(0,w,block)):
            patch=rgb[top:min(top+block,h),left:min(left+block,w)]
            if min(patch.shape[:2])<2:
                patch=rgb[max(0,h-block):h,max(0,w-block):w]
            if min(patch.shape[:2])<2:
                continue
            hf=(patch[:-1,:-1]-patch[1:,:-1]-patch[:-1,1:]+patch[1:,1:])*.5
            center=np.median(hf,axis=(0,1))
            grids[gy,gx]=np.median(np.abs(hf-center),axis=(0,1))/.67448975
    # Smooth tile estimates to avoid abrupt strength changes at block boundaries.
    grids=cv2.GaussianBlur(grids,(0,0),.8)
    maps=cv2.resize(grids,(w,h),interpolation=cv2.INTER_LINEAR)
    return np.clip(maps,0,.3).astype(np.float32)

def repair_outliers(rgb, sigma):
    median=cv2.medianBlur(np.rint(np.clip(rgb,0,1)*255).astype('uint8'),3).astype(np.float32)/255
    delta=np.abs(rgb-median)
    # Extreme isolated deviations only. Original pixels remain a network input.
    threshold=np.maximum(4*sigma,.20)
    weight=np.clip((delta-threshold)/np.maximum(threshold,.02),0,1)
    return rgb+weight*(median-rgb)

def wavelet_estimate(rgb: np.ndarray, sigma=None, strength=1.0, repair=True):
    sigma=noise_maps(rgb) if sigma is None else sigma
    clean=repair_outliers(rgb,sigma) if repair else rgb
    if min(rgb.shape[:2])<8:
        return clean.copy()
    color=clean@OPPONENT.T
    variance=(sigma*sigma)@(OPPONENT*OPPONENT).T
    result=np.empty_like(color)
    for c in range(3):
        wavelet=pywt.Wavelet('db2')
        level=min(3,pywt.dwt_max_level(min(color.shape[:2]),wavelet.dec_len))
        coeffs=pywt.wavedec2(color[...,c],wavelet,mode='symmetric',level=level)
        den=[coeffs[0]]
        for detail in coeffs[1:]:
            nv=cv2.resize(variance[...,c],(detail[0].shape[1],detail[0].shape[0]))
            shrunk=[]
            for arr in detail:
                total=cv2.blur(arr*arr,(5,5))
                signal=np.sqrt(np.maximum(total-nv,1e-8))
                threshold=strength*nv/(signal+1e-8)
                shrunk.append(np.sign(arr)*np.maximum(np.abs(arr)-threshold,0))
            den.append(tuple(shrunk))
        result[...,c]=pywt.waverec2(den,wavelet,mode='symmetric')[:rgb.shape[0],:rgb.shape[1]]
    return np.clip(result@OPPONENT,0,1).astype(np.float32)

def features(rgb: np.ndarray):
    sigma=noise_maps(rgb)
    base=wavelet_estimate(rgb,sigma)
    # Consistent quantization allows a compact, memory-mapped training cache.
    base=np.rint(base*255)/255
    sigma=np.rint(sigma/.3*255)/255*.3
    return np.concatenate([rgb,base,sigma],axis=-1).astype(np.float32)
