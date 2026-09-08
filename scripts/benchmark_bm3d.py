"""Optional BM3D classical comparison on a fixed, severity-spanning val subset."""
import json
import time
import numpy as np
import cv2
import bm3d
from common import SCRIPTS,public_root,load_split,read_rgb,to_uint8,save_rgb,write_json
from classical import noise_maps,repair_outliers
from metrics import score_arrays,summarize

def main():
    audit={r['id']:r for r in json.loads((SCRIPTS/'reports'/'audit.json').read_text())['rows']}
    ordered=sorted(load_split('val'),key=lambda i:audit[i]['noisy_sigma'])
    ids=[ordered[int(i)] for i in np.linspace(0,len(ordered)-1,8)]
    out=SCRIPTS/'runs'/'bm3d_val_subset'
    if (out/'metrics.json').exists():raise SystemExit('Benchmark already exists')
    cv2.setNumThreads(1);profile=bm3d.BM3DProfile();profile.num_threads=4
    rows=[]
    for id_ in ids:
        y=read_rgb(public_root()/'noisy'/f'{id_}_noise.png');gt=read_rgb(public_root()/'ground_truth'/f'{id_}.png')
        start=time.perf_counter();maps=noise_maps(y)
        corrected=repair_outliers(y,maps)
        sigma=float(np.median(maps))
        pred=bm3d.bm3d_rgb(corrected,sigma,profile=profile)
        elapsed=time.perf_counter()-start;pred=to_uint8(pred).astype('float32')/255
        save_rgb(out/'images'/f'{id_}.png',pred)
        rows.append(dict(id=id_,sigma=sigma,seconds=elapsed,**score_arrays(y,pred,gt)))
        print(id_,round(elapsed,2),round(rows[-1]['composite_score'],5),flush=True)
    comparators={}
    for name,path in [('baseline',SCRIPTS/'runs'/'baseline_val'/'metrics.json'),
                      ('wavelet',SCRIPTS/'runs'/'wavelet_val_1'/'metrics.json'),
                      ('hybrid16_4000',SCRIPTS/'runs'/'hybrid16'/'val_004000.json')]:
        other=json.loads(path.read_text())['rows']
        comparators[name]=summarize([r for r in other if r['id'] in ids])
    write_json(out/'metrics.json',dict(method='BM3D RGB + adaptive outlier repair',threads=4,ids=ids,
        summary=summarize(rows),rows=rows,same_image_comparisons=comparators))
    print(json.dumps(dict(bm3d=summarize(rows),**comparators),indent=2),flush=True)

if __name__=='__main__':main()
