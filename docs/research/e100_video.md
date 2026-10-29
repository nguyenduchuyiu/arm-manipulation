# e100 point-query XYZ videos

Status: planned; generated after the pilot's opening-selected checkpoint.
Native TCOW RGB+frame0dot, one top view, full finetune on512e96query episodes,
1500updates, batch4/micro2, backboneLR1e-5/headLR3e-4,XYZloss only.
Original pretrained TCOW backbone; no ID/GT XYZ/mask input.

Two saved-video localization cases: first validation episode and first both-unseen
test episode. Exact checkpoint/step, paths and runtime settings in config.json.
Each prediction uses30past/current frames, always retaining frame0. Radius3query
disk at frame0 only. Cached source25Hz/stride3plus final, nominal output25/3FPS;
raw source frame index supplies displayed physical time. Top center-cropped
240x320RGB, green measured center, magenta prediction and recent trajectories;
orange initial query. XYZ and error shown in base-frame metres/mm. Camera
projection is visualization only. No learned robot control, IK/actions,H/K,
flow steps or action-noise seed. Videos and coordinate NPZ have matching frames.
