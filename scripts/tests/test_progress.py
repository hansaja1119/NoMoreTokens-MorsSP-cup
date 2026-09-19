"""Never announce shutdown readiness for a single completed training phase."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from status import completion_state


def test_completion_requires_both_phases_and_final_artifacts(tmp_path):
    queue={'state':'complete'}
    assert completion_state(queue,{'state':'running'},tmp_path)[0]=='running'
    output=tmp_path/'outputs';output.mkdir();checkpoint=tmp_path/'model.pt'
    final=dict(state='complete',output=str(output),checkpoint=str(checkpoint))
    assert completion_state(queue,final,tmp_path)[0]=='attention'
    (output/'NoMoreTokens.zip').write_bytes(b'fixture');checkpoint.write_bytes(b'fixture')
    assert completion_state(queue,final,tmp_path)[0]=='complete'
    large=tmp_path/'large_width_experiments';large.mkdir()
    (large/'status.json').write_text('{"state":"running"}')
    assert completion_state(queue,final,tmp_path)[0]=='running'
    (large/'status.json').write_text('{"state":"failed","error":"memory probe failed"}')
    assert completion_state(queue,final,tmp_path)[0]=='attention'
    (large/'status.json').write_text('{"state":"paused","saved_step":8750}')
    assert completion_state(queue,final,tmp_path)[0]=='paused'
    (large/'status.json').write_text('{"state":"complete"}')
    assert completion_state(queue,final,tmp_path)[0]=='complete'
    assert completion_state({'state':'failed','error':'example'},final,tmp_path)[0]=='attention'
    extra=tmp_path/'width_experiments';extra.mkdir()
    (extra/'status.json').write_text('{"state":"running"}')
    assert completion_state(queue,final,tmp_path)[0]=='running'
    (extra/'status.json').write_text('{"state":"complete"}')
    assert completion_state(queue,final,tmp_path)[0]=='complete'
