"""Replay one pick episode with a calibrated, vertically downward RGB-D camera."""
from pathlib import Path

from OATFlow.data_tools.rear_episode import main

if __name__=='__main__':
    main(Path('/workspace/robotics_assets/experiments/e128_top'),
         Path('docs/research/e128_README.md'),view='top')
