# Minimal runnable code cleanup

Status: complete.

User requested extracting shared helpers and deleting old experiment artifacts,
prioritizing minimal runnable code. workspace_analysis and obsolete RGB3/ID
tracking/FM4D wrappers were removed. Current packages:tracking,reach,localization,
simulation,data_tools. Original checkpoint/data values and corrected videos kept.

Verification:four retained model predictions exactly match pre-refactor outputs;
189FMparameter order preserved;two resumed updates vsone+save/resume+one produce
identical model/AdamW/RNG;point full-finetune gradient reaches all121850888parameters;
400live tracking/FM/probe commands and full qpos match saved GreenCube rollout
exactly,holding bilateral contact and lifting79.659mm. Ninepoint tests passed;
all11CLI entry points import/parse.

Data smoke:8new voxel trajectories,4physical replays(maxEE mismatch3.1e-8m),
190EEpose chunks and aligned joint inputs,all audited. Shuffle pilot1context/4queries
collected anew,RGB3/masks rendered,labels/occlusion/splits audited;all4expert command
replays bit-exact. Full datasets not regenerated. Minimal FM trainer has no
validation/best-selection stage;current original benchmark results are unchanged.

Current runnable code snapshot replaces obsolete historical source archive.
Metadata report:robotics_assets/metadata/code_cleanup.json. All temporary scripts,
training checkpoints,test datasets and logs deleted after verification. Preserved
independent prior/RL project work and shared expert/environment/controllers.
Commands:OATFlow/{tracking,reach,localization,data_tools}/README.md.
