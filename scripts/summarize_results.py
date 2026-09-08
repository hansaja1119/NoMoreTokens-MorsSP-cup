"""Update the development scorecard and scene-group bootstrap intervals."""
import json
import numpy as np
from common import SCRIPTS,write_json

def main():
    baseline=json.loads((SCRIPTS/'runs'/'baseline_val'/'metrics.json').read_text())
    base={r['id']:r['composite_score'] for r in baseline['rows']}
    split=json.loads((SCRIPTS/'data'/'split.json').read_text())
    ids=split['val'];used=set();groups=[]
    for g in split['groups']:
        relevant=[i for i in g if i in ids]
        if relevant:groups.append(relevant);used.update(relevant)
    groups.extend([[i] for i in ids if i not in used])
    rng=np.random.default_rng(20260908)
    samples=rng.integers(0,len(groups),(10000,len(groups)))
    counts=np.array([len(g) for g in groups])
    entries=[]
    paths=[SCRIPTS/'runs'/'baseline_val'/'metrics.json',SCRIPTS/'runs'/'wavelet_val_1'/'metrics.json']
    paths+=sorted((SCRIPTS/'runs').glob('*/val_*.json'))
    for path in paths:
        result=json.loads(path.read_text());rows=result['rows']
        if {r['id'] for r in rows}!=set(ids):continue
        model={r['id']:r['composite_score'] for r in rows}
        group_sums=np.array([sum(model[i]-base[i] for i in g) for g in groups])
        values=group_sums[samples].sum(1)/counts[samples].sum(1)
        ci=np.quantile(values,[.025,.975]).tolist()
        entries.append(dict(run=path.parent.name,step=result.get('step'),summary=result['summary'],paired_gain_ci95=ci,path=str(path)))
    write_json(SCRIPTS/'reports'/'scorecard.json',dict(validation_images=len(ids),bootstrap_groups=len(groups),replicates=10000,entries=entries))
    lines=['# VISION_HUNTERS development scorecard','',
           'All quality values below are from the same 60-image validation split. The locked test remains reserved.',
           f'Gain intervals resample {len(groups)} scene groups, using 10,000 paired bootstrap replicates.',
           'They quantify uncertainty within this dataset, not the probability of winning.',
           '', '| Run | Steps | PSNR | SSIM | Composite | Gain over baseline, 95% interval |',
           '|---|---:|---:|---:|---:|---|']
    for r in entries:
        m=r['summary'];lo,hi=r['paired_gain_ci95']
        lines.append(f'| {r["run"]} | {r["step"] or "-"} | {m["psnr"]:.3f} | {m["ssim"]:.4f} | {m["composite_score"]:.5f} | [{lo:.4f}, {hi:.4f}] |')
    lines+=['','Runtime values in individual logs use different devices and concurrency. Do not compare their runtime columns as an isolated CPU benchmark.',
            'The prototype processed all 20 preliminary images on CPU in 66.6 seconds while other development work ran.',
            'The development ZIP and PDF are not a frozen submission. Longer comparisons and final training are still required.']
    (SCRIPTS/'reports'/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines))

if __name__=='__main__':main()
