import importlib.util
from pathlib import Path
import sys
import json
import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import SCRIPTS,ROOT,to_uint8,read_rgb
from classical import noise_maps,wavelet_estimate,features
from denoise import output_name
from model import Denoiser
from inference import predict,protect_mild,weak_texture_luma_sigma
from metrics import official,score_arrays

def test_scene_groups_do_not_leak():
    split=json.loads((SCRIPTS/'data'/'split.json').read_text())
    sets=[set(split[k]) for k in ['train','val','test']]
    assert [len(s) for s in sets]==[340,60,60]
    assert len(set.union(*sets))==460
    assert not any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
    for group in json.loads((SCRIPTS/'data'/'scene_groups.json').read_text())['groups']:
        assert sum(bool(set(group)&s) for s in sets)==1

def test_official_metric_parity_after_png(tmp_path):
    rng=np.random.default_rng(21)
    gt=rng.uniform(.2,.8,(32,31,3)).astype('float32')
    y=np.clip(gt+rng.normal(0,.08,gt.shape),0,1).astype('float32')
    pred=(gt+y)*.5
    for name,x in [('001',gt),('001_noise',y),('pred',pred)]:
        Image.fromarray(to_uint8(x)).save(tmp_path/f'{name}.png')
    row=official.evaluate_image('001',tmp_path/'pred.png',tmp_path/'001.png',tmp_path/'001_noise.png')
    ours=score_arrays(read_rgb(tmp_path/'001_noise.png'),read_rgb(tmp_path/'pred.png'),read_rgb(tmp_path/'001.png'))
    for key in ours:assert ours[key]==pytest.approx(row[key],abs=1e-9)
    assert official.official_composite(30,.2)==pytest.approx(.68)
    assert official.official_composite(-1,-.2)==0

def test_adaptive_classical_removes_noise_without_changing_clean_constant():
    x=np.full((129,117,3),.4,np.float32)
    assert np.max(np.abs(wavelet_estimate(x)-x))<1e-5
    rng=np.random.default_rng(42)
    y=np.clip(x+rng.normal(0,.06,x.shape),0,1).astype('float32')
    out=wavelet_estimate(y)
    assert np.mean((out-x)**2)<.4*np.mean((y-x)**2)

@pytest.mark.parametrize('shape',[(1,1),(7,11),(63,65),(129,117)])
def test_dimensions_and_tile_edges(shape):
    torch.set_num_threads(2)
    y=np.random.default_rng(1).uniform(0,1,(*shape,3)).astype('float32')
    model=Denoiser(width=4,hybrid=False).eval()
    out=predict(model,y,torch.device('cpu'),tile=64,overlap=16)
    assert out.shape==y.shape
    assert np.isfinite(out).all()
    assert np.max(np.abs(out-y))<1e-5

def test_filename_contract_and_finite_output():
    assert output_name(Path('481_noise.png'))=='481.png'
    assert output_name(Path('my_photo.jpg'))=='my_photo.png'
    assert output_name(Path('noise_picture_noise.PNG'))=='noise_picture.png'
    with pytest.raises(ValueError):to_uint8(np.array([np.nan]))

def test_training_update_is_finite():
    torch.set_num_threads(2);torch.manual_seed(1)
    m=Denoiser(width=4,hybrid=False)
    x=torch.rand(2,3,32,32)
    target=torch.nn.functional.avg_pool2d(x,3,1,1)
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    before=torch.nn.functional.mse_loss(m(x),target).item()
    for _ in range(5):
        opt.zero_grad();loss=torch.nn.functional.mse_loss(m(x),target);loss.backward();opt.step()
        assert torch.isfinite(loss)
    assert torch.nn.functional.mse_loss(m(x),target).item()<before

def test_mild_protection_does_not_ignore_correlated_noise():
    rng=np.random.default_rng(36)
    x=np.full((128,128,3),.5,np.float32)
    y=np.clip(x+rng.normal(0,.1,(128,128,1)),0,1).astype('float32')
    assert weak_texture_luma_sigma(y)>.05
    guarded=protect_mild(y,x,noise_maps(y),.03)
    assert np.mean((guarded-x)**2)<1e-6
    altered=x+.02
    assert np.array_equal(protect_mild(x,altered,noise_maps(x),.03),x)
