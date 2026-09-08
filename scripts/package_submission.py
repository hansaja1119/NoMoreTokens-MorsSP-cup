"""Validate and package only the 20 required PNG outputs; never upload anything."""
import argparse
import json
from pathlib import Path
import zipfile
from PIL import Image
from common import SCRIPTS,sha256,write_json

def validate_images(directory):
    expected={f'{i:03d}.png' for i in range(461,481)}
    found={p.name for p in directory.iterdir() if p.is_file()}
    if found!=expected:raise ValueError(f'Expected exactly 20 outputs; missing={sorted(expected-found)}, extra={sorted(found-expected)}')
    hashes={}
    for name in sorted(expected):
        path=directory/name
        with Image.open(path) as im:
            if im.format!='PNG' or im.size!=(992,992) or im.mode!='RGB':
                raise ValueError(f'{name}: require 992x992 RGB PNG')
            im.verify()
        hashes[name]=sha256(path)
    return hashes

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--images',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=SCRIPTS/'outputs'/'submission')
    ap.add_argument('--inference-manifest',type=Path,required=True)
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--commit',required=True,help='Exact code revision; do not invent a placeholder SHA')
    args=ap.parse_args()
    import re
    if not re.fullmatch('[0-9a-f]{40}',args.commit):ap.error('Expected a full 40-character Git commit SHA')
    hashes=validate_images(args.images)
    manifest=json.loads(args.inference_manifest.read_text())
    if hashes!={r['output']:r['sha256'] for r in manifest['images']}:raise ValueError('Images do not match inference manifest')
    if manifest['checkpoint_sha256']!=sha256(args.checkpoint):raise ValueError('Checkpoint checksum mismatch')
    team=json.loads((SCRIPTS/'data'/'team.json').read_text())['team_name']
    args.output_dir.mkdir(parents=True,exist_ok=True)
    target=args.output_dir/f'{team}.zip'
    if target.exists():raise FileExistsError(target)
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in sorted(hashes):z.write(args.images/name,arcname=name)
    with zipfile.ZipFile(target) as z:
        if set(z.namelist())!=set(hashes) or z.testzip() is not None:raise ValueError('ZIP verification failed')
    write_json(args.output_dir/'submission_manifest.json',dict(team=team,code_commit=args.commit,
        checkpoint_sha256=sha256(args.checkpoint),zip_sha256=sha256(target),images=hashes))
    print(target)

if __name__=='__main__':main()
