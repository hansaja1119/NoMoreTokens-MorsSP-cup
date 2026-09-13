"""Exercise the final-training branch without touching actual held-out data."""
from pathlib import Path
import sys
import json
import numpy as np
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import common
import train
import training_data

def test_all_data_training_requires_locked_record_and_disables_validation(tmp_path,monkeypatch):
    for mod in [common,train,training_data]:monkeypatch.setattr(mod,'SCRIPTS',tmp_path)
    data=tmp_path/'data';data.mkdir()
    split=dict(train=['001','002'],val=['003'],test=['004'])
    (data/'split.json').write_text(json.dumps(split))
    (tmp_path/'classical.py').write_text('# test-only preprocessing marker\n')
    cache=tmp_path/'cache'/'features_v1';cache.mkdir(parents=True)
    for id_ in ['001','002','003','004']:
        np.save(cache/f'{id_}.npy',np.full((256,256,12),100,np.uint8))
    (cache/'manifest.json').write_text(json.dumps(dict(ids=['001','002','003','004'],
        classical_sha256=common.sha256(tmp_path/'classical.py'),split_sha256=common.sha256(data/'split.json'))))
    argv=['train.py','--run','smoke','--steps','1','--batch','1','--accumulate','1',
          '--workers','0','--width','4','--device','cpu','--no-synthetic','--final-training']
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(RuntimeError,match='one-time locked test'):train.main()
    record=tmp_path/'runs'/'final_preparation'/'locked_test'/'metrics.json'
    record.parent.mkdir(parents=True);record.write_text('{}')
    def no_validation(*args,**kwargs):raise AssertionError('All-data training must not evaluate the held-out partitions')
    monkeypatch.setattr(train,'evaluate',no_validation)
    train.main()
    result=torch.load(tmp_path/'runs'/'smoke'/'final.pt',map_location='cpu',weights_only=True)
    assert result['step']==1
    assert result['validation_enabled'] is False
    assert result['best_score'] is None
    assert set(training_data.TrainingCrops(1,1,final_training=True).ids)=={'001','002','003','004'}
    assert not list((tmp_path/'runs'/'smoke').glob('val_*.json'))
    # Inference exports contain EMA only: initialization must not eagerly read
    # the absent raw-model/optimizer keys (exact --resume still needs them).
    exported=tmp_path/'inference_only.pt'
    torch.save({key:result[key] for key in ['model_config','ema','step','split_sha256']},exported)
    init_argv=argv.copy();init_argv[init_argv.index('smoke')]='initialized'
    monkeypatch.setattr(sys,'argv',init_argv+['--init',str(exported)])
    train.main()
    initialized=torch.load(tmp_path/'runs'/'initialized'/'final.pt',map_location='cpu',weights_only=True)
    assert initialized['step']==1 and initialized['validation_enabled'] is False
