"""Evaluate final E121 on the same3200held-out K10 pick episodes."""
import argparse
from OATFlow.top_pick.cloud_evaluate import main
from OATFlow.top_pick.patch_fusion import ROOT,PatchFusionPolicy

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',action='store_true')
    main(parser.parse_args().resume,root=ROOT,policy_class=PatchFusionPolicy,readme='docs/research/e121_test_README.md')
