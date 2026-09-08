"""Reject incomplete or inconsistent selection before opening held-out data."""
from pathlib import Path
import sys
import json
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import sha256
from prepare_final_candidate import verify_selection


def test_promotion_requires_matching_runtime_and_passing_measurements(tmp_path):
    selected=tmp_path/'selected.pt';selected.write_bytes(b'test checkpoint identity')
    selection=dict(checkpoint_sha256=sha256(selected),mild_threshold=.03)
    status=tmp_path/'status.json';status.write_text(json.dumps(dict(state='running')))
    with pytest.raises(RuntimeError,match='not complete'):verify_selection(selection,selected,tmp_path)
    status.write_text(json.dumps(dict(state='complete')))
    runtime=dict(checkpoint_sha256='different',cpu_repeat_png_exact=True,cpu_gpu_max_png_difference=1)
    runtime_path=tmp_path/'selected_runtime.json';runtime_path.write_text(json.dumps(runtime))
    with pytest.raises(RuntimeError,match='different checkpoint'):verify_selection(selection,selected,tmp_path)
    runtime['checkpoint_sha256']=selection['checkpoint_sha256'];runtime_path.write_text(json.dumps(runtime))
    checks=[dict(threshold=.03,passes=True,diagnostics=dict(clean=dict(mae=0),
                mild_gaussian=dict(input_mse=.0001,output_mse=.0002)))]
    robust=tmp_path/'robustness_selection.json';robust.write_text(json.dumps(checks))
    with pytest.raises(RuntimeError,match='measurements'):verify_selection(selection,selected,tmp_path)
    checks[0]['diagnostics']['mild_gaussian']['output_mse']=.00009;robust.write_text(json.dumps(checks))
    verify_selection(selection,selected,tmp_path)
