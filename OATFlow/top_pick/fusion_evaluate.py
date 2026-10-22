"""Full held-out evaluation of the final RGB/scene-depth fusion policy."""
import argparse
from OATFlow.top_pick.fusion import ROOT,FusionPolicy
from OATFlow.top_pick.cloud_evaluate import main

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    main(parser.parse_args().resume,root=ROOT,policy_class=FusionPolicy,readme='docs/research/e120_test_README.md')
