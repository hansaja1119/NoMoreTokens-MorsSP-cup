"""Shared paths, image I/O and durable experiment metadata."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

SCRIPTS = Path(__file__).resolve().parent
ROOT = SCRIPTS.parent

def public_root() -> Path:
    canonical = ROOT / 'competition_data' / 'public'
    return canonical if (canonical / 'noisy').is_dir() else ROOT / 'public'

def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.float32) / 255.0

def to_uint8(x: np.ndarray) -> np.ndarray:
    if not np.isfinite(x).all():
        raise ValueError('Prediction contains non-finite pixels')
    return np.rint(np.clip(x, 0, 1) * 255).astype(np.uint8)

def save_rgb(path: Path, x: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(x)).save(path)

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')
    tmp.replace(path)

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def load_split(name: str) -> list[str]:
    data = json.loads((SCRIPTS / 'data' / 'split.json').read_text())
    if name == 'all':
        return sorted(data['train'] + data['val'] + data['test'])
    return data[name]
