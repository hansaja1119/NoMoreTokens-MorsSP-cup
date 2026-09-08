"""Read-only dataset measurements; writes analysis and a scene-grouped split.

No model parameters are fit from validation/test residuals. Noise synthesis
calibration is exported separately using training pairs only.
"""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import time
import numpy as np
from PIL import Image, ImageDraw
from common import SCRIPTS, public_root, read_rgb, write_json

def inspect(image_id):
    p = public_root()
    x = read_rgb(p / 'ground_truth' / f'{image_id}.png')
    y = read_rgb(p / 'noisy' / f'{image_id}_noise.png')
    if x.shape != y.shape or x.shape != (992, 992, 3):
        raise ValueError(f'{image_id}: unexpected pair dimensions')
    r = y - x
    # Decimation reduces audit cost, without modifying any source image.
    xs, ys, rs = x[::3,::3], y[::3,::3], r[::3,::3]
    lum = x.mean(-1)
    noisy_lum = y.mean(-1)
    hf = noisy_lum[:-1,:-1] - noisy_lum[1:,:-1] - noisy_lum[:-1,1:] + noisy_lum[1:,1:]
    mad = float(np.median(np.abs(hf - np.median(hf))) / (0.67448975 * 2))
    sigma = float(np.std(rs[np.abs(rs) < .15]))
    rc = np.clip(r, -.15, .15)
    rc -= rc.mean(axis=(0,1), keepdims=True)
    var = np.mean(rc * rc)
    row = dict(id=image_id, brightness=float(x.mean()), noisy_brightness=float(y.mean()),
               noisy_sigma=mad, psnr=float(-10*np.log10(np.mean(r*r))),
               residual_mean=float(r.mean()), residual_std=float(r.std()),
               trimmed_std=sigma, outlier_fraction=float(np.mean(np.abs(rs)>.25)),
               residual_corr_h=float(np.mean(rc[:,1:]*rc[:,:-1])/max(var,1e-12)),
               residual_corr_v=float(np.mean(rc[1:]*rc[:-1])/max(var,1e-12)),
               row_bias_std=float(rc.mean(axis=1).std()),
               col_bias_std=float(rc.mean(axis=0).std()),
               noisy_black_fraction=float(np.mean(ys==0)),
               noisy_white_fraction=float(np.mean(ys==1)),
               bias_rgb=[float(z) for z in r.mean(axis=(0,1))])
    bins=[]
    for lo,hi in zip([0,.05,.1,.2,.4,.7],[.05,.1,.2,.4,.7,1.01]):
        selected = rs[(xs>=lo)&(xs<hi)]
        bins.append({'lo':lo,'hi':hi,'n':int(selected.size),
                     'mean':float(selected.mean()) if selected.size else 0.,
                     'var':float(selected.var()) if selected.size else 0.})
    row['intensity_bins']=bins
    thumb = np.asarray(Image.fromarray((lum*255).astype('uint8')).resize((32,32)),dtype=np.float32)/255
    dh = np.asarray(Image.fromarray((lum*255).astype('uint8')).resize((9,8)))
    dh = (dh[:,1:]>dh[:,:-1]).flatten()
    digest=hashlib.sha256((x*255).astype('uint8').tobytes()).hexdigest()
    return row,thumb,dh,digest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workers',type=int,default=4)
    args=ap.parse_args()
    started=time.perf_counter()
    ids=[f'{i:03d}' for i in range(1,461)]
    rows=[]; thumbs=[]; dh=[]; hashes=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,(row,t,d,h) in enumerate(pool.map(inspect,ids)):
            rows.append(row);thumbs.append(t);dh.append(d);hashes.append(h)
            if (i+1)%40==0: print(f'Audited {i+1}/460',flush=True)
    thumbs=np.stack(thumbs);dh=np.stack(dh)
    parents=list(range(len(ids)))
    def find(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    pairs=[]
    for i in range(len(ids)):
        distances=np.sum(dh[i+1:]!=dh[i],axis=1)
        for j in np.where(distances<=10)[0]+i+1:
            mse=float(np.mean((thumbs[i]-thumbs[j])**2))
            same=hashes[i]==hashes[j]
            if same or (distances[j-i-1]<=8 and mse<.004):
                parents[find(j)]=find(i)
                pairs.append({'a':ids[i],'b':ids[j],'thumb_mse':mse,'exact':same})
    reviewed=json.loads((SCRIPTS/'data'/'scene_groups.json').read_text())['groups']
    for group in reviewed:
        first=ids.index(group[0])
        for image_id in group[1:]:parents[find(ids.index(image_id))]=find(first)
    groups={}
    for i in range(len(ids)):groups.setdefault(find(i),[]).append(i)
    rng=np.random.default_rng(20260908)
    # Round-robin strata interleave dark/bright and weak/strong noisy observations.
    b=np.array([r['noisy_brightness'] for r in rows]);s=np.array([r['noisy_sigma'] for r in rows])
    strata={}
    for group in groups.values():
        key=(int(np.digitize(b[group].mean(),np.quantile(b,[1/3,2/3]))),
             int(np.digitize(s[group].mean(),np.quantile(s,[1/3,2/3]))))
        strata.setdefault(key,[]).append(group)
    split={'train':[],'val':[],'test':[]};targets={'train':340,'val':60,'test':60}
    for key in sorted(strata):
        gs=strata[key];rng.shuffle(gs)
        local={k:0 for k in split}
        for group in sorted(gs,key=len,reverse=True):
            choices=[k for k in split if len(split[k])+len(group)<=targets[k]] or list(split)
            name=min(choices,key=lambda k:(local[k]/targets[k],len(split[k])/targets[k]))
            split[name].extend(ids[i] for i in group);local[name]+=len(group)
    for k in split:split[k].sort()
    split.update(seed=20260908,grouping='Visual scene_groups.json plus exact RGB SHA256 or dHash distance <=8 and 32px grayscale MSE <0.004',
                 groups=[[ids[i] for i in g] for g in groups.values() if len(g)>1])
    split_path=SCRIPTS/'data'/'split.json'
    if split_path.exists() and json.loads(split_path.read_text())!=split:
        if (SCRIPTS/'runs').exists():
            raise RuntimeError('Refusing to change a split after experiments have started')
        write_json(SCRIPTS/'reports'/'split_before_visual_review.json',json.loads(split_path.read_text()))
    write_json(split_path,split)
    write_json(SCRIPTS/'reports'/'audit.json',{'rows':rows,'related_pairs':pairs})
    train_rows=[r for r in rows if r['id'] in split['train']]
    calibrate={k:np.quantile([r[k] for r in train_rows],[0,.1,.5,.9,1]).tolist()
               for k in ['trimmed_std','noisy_sigma','outlier_fraction','residual_mean']}
    write_json(SCRIPTS/'data'/'training_noise_statistics.json',{'source':'train only','quantiles':[0,.1,.5,.9,1],**calibrate})
    fields=[k for k in rows[0] if k not in ['intensity_bins','bias_rgb']]
    with (SCRIPTS/'reports'/'audit.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    text=['# Dataset audit','',f'460 public pairs; completed in {time.perf_counter()-started:.1f} seconds.',
          '',f'Split: {len(split["train"])} train / {len(split["val"])} validation / {len(split["test"])} locked test.',
          f'Automatic duplicate pairs: {len(pairs)}; conservative visual scene groups: {len(reviewed)}.',
          '', '| Measurement | Minimum | Median | Maximum |','|---|---:|---:|---:|']
    for k in fields[1:]:
        vals=[r[k] for r in rows]
        text.append(f'| {k} | {min(vals):.6f} | {np.median(vals):.6f} | {max(vals):.6f} |')
    text+=['','Noise-model parameters must use training_noise_statistics.json, not the all-image descriptive audit.',
           'Locked-test metrics are not used for model selection.']
    (SCRIPTS/'reports'/'AUDIT.md').write_text('\n'.join(text),encoding='utf-8')
    # Review sheets of all scenes, plus full-resolution crop comparisons.
    for page in range(4):
        part=ids[page*120:(page+1)*120]
        sheet=Image.new('RGB',(1200,((len(part)+9)//10)*130),'#202020');draw=ImageDraw.Draw(sheet)
        for n,id_ in enumerate(part):
            with Image.open(public_root()/'ground_truth'/f'{id_}.png') as im:
                im.thumbnail((116,108));xy=((n%10)*120,(n//10)*130)
                sheet.paste(im,xy);draw.text((xy[0],xy[1]+110),id_,fill='white')
        sheet.save(SCRIPTS/'reports'/f'scenes_{page+1}.jpg',quality=90)
    print('\n'.join(text[:8]),flush=True)
    print('Training-only statistics:',json.dumps(calibrate),flush=True)

if __name__=='__main__':main()
