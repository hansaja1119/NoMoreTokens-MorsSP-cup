from pathlib import Path
import json
import os
import subprocess
import sys
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import SCRIPTS
from model import Denoiser
from package_submission import validate_images
import pytest

def test_real_cli_rgb_gray_alpha_and_names(tmp_path):
    source=tmp_path/'noisy';source.mkdir()
    Image.fromarray(np.full((31,29,3),120,np.uint8)).save(source/'481_noise.png')
    Image.fromarray(np.full((7,9),80,np.uint8)).save(source/'gray.png')
    rgba=np.zeros((17,19,4),np.uint8);rgba[...,:3]=140;rgba[...,3]=77
    Image.fromarray(rgba).save(source/'alpha_noise.png')
    model=Denoiser(width=4,hybrid=False)
    checkpoint=tmp_path/'weights.pt'
    torch.save(dict(model_config=model.config,model=model.state_dict()),checkpoint)
    dest=tmp_path/'out';manifest=tmp_path/'manifest.json'
    cmd=[sys.executable,str(SCRIPTS/'denoise.py'),'--input_dir',str(source),'--output_dir',str(dest),
         '--checkpoint',str(checkpoint),'--device','cpu','--manifest',str(manifest)]
    process=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    assert process.returncode==0,process.stderr
    assert {p.name for p in dest.iterdir()}=={'481.png','gray.png','alpha.png'}
    with Image.open(dest/'481.png') as im:assert im.size==(29,31)
    with Image.open(dest/'gray.png') as im:assert im.size==(9,7)
    with Image.open(dest/'alpha.png') as im:assert np.all(np.asarray(im)[...,3]==77)
    assert len(json.loads(manifest.read_text())['images'])==3
    again=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
    assert again.returncode!=0 and 'Output exists' in again.stderr

def test_submission_validator_rejects_incomplete_and_wrong_shapes(tmp_path):
    with pytest.raises(ValueError,match='exactly 20'):validate_images(tmp_path)
    for n in range(461,481):Image.new('RGB',(8,8)).save(tmp_path/f'{n}.png')
    with pytest.raises(ValueError,match='992x992 RGB PNG'):validate_images(tmp_path)
