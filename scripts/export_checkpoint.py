"""Export local inference weights and provenance without optimizer state."""
import argparse
from pathlib import Path
import subprocess
import torch
from common import SCRIPTS,ROOT,sha256,write_json

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mild-threshold',type=float,default=.03)
    args=ap.parse_args()
    if args.output.exists():ap.error('Checkpoint exists; use a new versioned output filename')
    obj=torch.load(args.source,map_location='cpu',weights_only=True)
    export={k:obj[k] for k in ['model_config','ema','step','split_sha256']}
    export['inference_config']=dict(mild_threshold=args.mild_threshold)
    export['development_only']=True
    args.output.parent.mkdir(parents=True,exist_ok=True);torch.save(export,args.output)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    write_json(args.output.with_suffix('.json'),dict(source=str(args.source),source_sha256=sha256(args.source),
        checkpoint_sha256=sha256(args.output),code_revision=revision,development_only=True,
        inference_config=export['inference_config'],source_hashes={p.name:sha256(p) for p in SCRIPTS.glob('*.py')}))
    print(args.output)

if __name__=='__main__':main()
