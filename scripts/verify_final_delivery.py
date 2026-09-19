"""Verify delivered PNGs, package them, and create a viewable image inventory."""
import json
import statistics
import subprocess
import time
import zipfile
import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from common import ROOT, SCRIPTS, sha256, read_rgb, public_root, to_uint8, write_json
from inference import load_model, predict
from package_submission import validate_images


def main():
    out = ROOT / 'final_output'
    checkpoint = SCRIPTS / 'checkpoints' / 'NoMoreTokens_w128_s20000.pt'
    manifest = json.loads((out / 'inference_manifest.json').read_text())
    hashes = validate_images(out / 'images')
    assert hashes == {r['output']: r['sha256'] for r in manifest['images']}
    assert sha256(checkpoint) == manifest['checkpoint_sha256']
    for row in manifest['images']:
        row['input_sha256'] = sha256(ROOT / 'submissions' / 'noisy' / row['input'])
    write_json(out / 'inference_manifest.json', manifest)
    # Match the trained inference implementation to the archived source hashes.
    trained = json.loads((SCRIPTS / 'runs' / 'hybrid128' / 'config.json').read_text())
    for name in ['model.py', 'inference.py', 'classical.py', 'common.py']:
        assert sha256(SCRIPTS / name) == trained['source_hashes'][name], name
    runtime = out / 'runtime_checks.json'
    if not runtime.exists():
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        cpu = torch.device('cpu')
        model, metadata = load_model(checkpoint, cpu)
        y = read_rgb(public_root() / 'noisy' / '021_noise.png')
        start = time.perf_counter()
        a = to_uint8(predict(model, y, cpu))
        seconds = time.perf_counter() - start
        print(f'CPU first pass: {seconds:.2f}s', flush=True)
        b = to_uint8(predict(model, y, cpu))
        assert np.array_equal(a, b), 'CPU repeat mismatch'
        c = np.asarray(Image.open(SCRIPTS / 'outputs' / 'final_delivery' / 'validation' / 'images' / '021.png'))
        delta = int(np.abs(a.astype('int16') - c.astype('int16')).max())
        assert delta <= 1, f'CPU/CUDA difference {delta}'
        gpu = torch.device('cuda')
        model = model.to(gpu)
        repeated = to_uint8(predict(model, read_rgb(ROOT / 'submissions' / 'noisy' / '461_noise.png'), gpu))
        delivered = np.asarray(Image.open(out / 'images' / '461.png'))
        assert np.array_equal(repeated, delivered), 'Delivered CUDA output does not repeat'
        write_json(runtime, dict(checkpoint_sha256=sha256(checkpoint), width=metadata['model_config']['width'],
            step=metadata['step'], mild_threshold=metadata['inference_config']['mild_threshold'],
            validation_image='021', cpu_seconds=seconds, cpu_repeat_png_exact=True,
            cpu_gpu_max_png_difference=delta, delivered_image_repeat='461.png', cuda_repeat_png_exact=True,
            scope='One full validation image CPU twice and CUDA comparison; one delivered image CUDA repeat. No full-frame 1024px test.'))
    else:
        assert json.loads(runtime.read_text())['checkpoint_sha256'] == sha256(checkpoint)
    archive = out / 'NoMoreTokens.zip'
    if not archive.exists():
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for name in sorted(hashes):
                z.write(out / 'images' / name, arcname=name)
    with zipfile.ZipFile(archive) as z:
        assert sorted(z.namelist()) == sorted(hashes) and z.testzip() is None
        import hashlib
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in hashes.items())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    write_json(out / 'submission_manifest.json', dict(team='NoMoreTokens', code_base_commit=revision,
        provenance_note='Base commit before handoff; exact working-tree inference code identified by source hashes below.',
        source_hashes={n: sha256(SCRIPTS / n) for n in ['model.py', 'inference.py', 'classical.py', 'common.py', 'denoise.py']},
        checkpoint_sha256=sha256(checkpoint), zip_sha256=sha256(archive), images=hashes))
    sheet = Image.new('RGB', (1100, 4 * 248), '#eeeeee')
    draw = ImageDraw.Draw(sheet)
    lines = ['# Final denoised image list', '',
        '20 RGB PNG images, each 992 × 992, generated from width 128 / 20,000-step EMA / mild threshold 0.03 using CUDA.', '',
        'Download `NoMoreTokens.zip` from the GitHub release for the image-only competition archive. All files are also in [images](images).', '',
        'No clean preliminary targets are supplied; no quality score is claimed for these 20 images.', '',
        '| Noisy input | Denoised output | Bytes | SHA-256 |', '|---|---|---:|---|']
    for idx, row in enumerate(manifest['images']):
        p = out / 'images' / row['output']
        with Image.open(p) as im:
            thumb = im.resize((208, 208), Image.Resampling.LANCZOS)
        x, y = (idx % 5)*220+6, (idx//5)*248+6
        sheet.paste(thumb, (x, y))
        draw.text((x, y+215), row['output'], fill='#111111')
        lines.append(f'| {row["input"]} | [{p.name}](images/{p.name}) | {p.stat().st_size:,} | `{row["sha256"]}` |')
    sheet.save(out / 'contact_sheet.jpg', quality=90)
    lines += ['', '![All 20 denoised images](contact_sheet.jpg)', '',
        f'Total PNG bytes: {sum((out/"images"/n).stat().st_size for n in hashes):,}. ZIP bytes: {archive.stat().st_size:,}.',
        f'ZIP SHA-256: `{sha256(archive)}`.', '',
        '[Inference manifest](inference_manifest.json) · [Submission manifest](submission_manifest.json) · [Runtime checks](runtime_checks.json)', '']
    (out / 'IMAGE_LIST.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Verified {len(hashes)} images, archive hashes, and repeatability.', flush=True)


if __name__ == '__main__':
    main()
