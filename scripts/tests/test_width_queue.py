from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_width_experiments import qualifies


def test_only_strictly_better_4000_step_scores_qualify():
    assert qualifies(.565,.5646054497225644)
    assert not qualifies(.5646054497225644,.5646054497225644)
    assert not qualifies(.56,.5646054497225644)
