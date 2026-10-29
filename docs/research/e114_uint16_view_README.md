# E114 decoded uint16 depth visualization

Status: complete. Visualize the actual compressed FFV1 depth file, not newly
rendered float32 depth. Input test_depth_compress/depth_uint16_0p1mm.mkv,
621frames,320x240,25FPS,unit0.0001m,code0 invalid. File decoded as gray16le;
metres recovered as uint16*0.0001. Invalid background pixels displayed dark.
Fixed turbo_r0.5--1.25m color scale. Native source is ideal MuJoCo depth.

Output pairs regenerated RGB+saved TCOW mask with decoded colorized depth and
the same saved target point cloud. Target cloud is reused from original replay,
not recomputed or retracked using quantized depth. Cyan measured/orange inferred
with white outline. Same camera/crop/perspective; no naive comparison or scene
point lattice. Orange cloud is shown through cover as an explicit hypothesis.

Case g000/p01/t1 BlueCube,24.84s,recorded expert reveal/cover/shuffle/uncover/lift.
TCOW e114_mask/train_3/best.pt epoch9/step4608 unchanged; no training/inference.
No new epochs/batch/LR/H/K/flow/noise settings. Original training metadata in run.
Prior geometry limitations: e114_front_README.md. Quantization/codec measurements:
e114_depth_compress_README.md. MP4 is a colorized preview, not the metric storage
format; the original16-bit MKV remains the numeric data source.

Checks: decoded input frame count, source depth SHA256 unchanged, full output
decode, camera roundtrip on valid pixels, source mask/cloud/metric hashes and
visual inspection. Resolution1440x860,621frames,25FPS.

Run: `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.tracking.pointcloud_camera_video`
Output: `/workspace/robotics_assets/experiments/e114_mask/test_depth_compress/view_uint16`.

Output fully decoded,621frames at25FPS. Input FFV1 SHA256 unchanged;
source mask/cloud/metric hashes unchanged. Valid-pixel camera roundtrip
1.14e-13pixels. Initial screenshot visually inspected, including dark invalid
background and restored target cloud. The target overlay remains the original
saved geometry; only depth background is supplied by the decoded uint16 file.
