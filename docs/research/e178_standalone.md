# E178 code-only reproducible bundle

Location: `/workspace/arm-manipulation/unified_geometry_semantic`.
Selected latest E178, original train source commit
ed39d076aa330262ed0df6a91ce2ec5c60262886. Original experiments untouched.

The initial extraction used frozen input datasets. The user's final requirement
is code-driven regeneration without shipping datasets. The final bundle includes
E135 procedural pose sampling, E157 physical visual/collision resize and RGB-D
rendering, E175 shape/camera renderers, UV graph construction, original E137/E169
SE(3) RNG transforms, pinned DINO downloader, extraction, scratch training and all
evaluation/control/report stages. Scene XML, robot meshes and initial robot/camera
configuration are required source assets (~22MiB), not a sampled dataset.

`run.py full` performs fetch_weights → generate → generate_tests → cache →
preflight → train → evaluate → ood → controls → report. No runtime imports of
OATFlow or other experiment code. Reference JSONs contain historical metrics only;
new observations/calibration/graphs/poses are generated. Model, training schedule,
EMD objective, core metrics and SE(3) helpers remain unchanged from E178.
Backbone weights downloaded from a pinned HF revision with SHA verification.

Generation source hashes: reference/generation_sources.json. Actual regeneration
verification: reference/generation_validation.json. Model/gradient/SE(3) smoke
verification: reference/bundle_smoke.json. MANIFEST.json covers source/assets.
Generated test datasets/caches and copied original datasets/checkpoints are excluded
from delivery. Runtime artifacts go under runs/; environment under .venv/.
No full30k retrain was launched; successful data regeneration and model smoke do
not constitute a new model accuracy measurement. Historical accuracy in
reference/RESULTS.md must not be presented as a retrain result.

E178 semantic conditioning is limited to four color task IDs. E177 block6 is
better on all four heldout shape families despite E178 being newer. One seed,
ideal depth, fixed clutter, visible surface only. See bundle README.md for exact
commands, dependencies and scope.
