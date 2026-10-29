# E129 calibrated front/top point-cloud fusion

Status: COMPLETE.

Input: synchronized E128 RGB-D, calibration, g000/p00/t0, 138 frames at 25 Hz.
Output: /workspace/robotics_assets/experiments/e129_fuse.
Back-project optical-Z depth (uint16, 2 mm/unit) with each camera's K and
camera-to-base transform, crop to tabletop/robot ROI, concatenate both clouds,
and compute XYZ centroid plus mean RGB in fixed 3 mm robot-base voxels.
Keep per-view observation counts for every voxel. Fusion needs no mask,
training, checkpoint, optimizer, epochs, batch, learning rate, actions, H/K
or flow settings. No ICP, TSDF, temporal accumulation or surface completion.
GT instance masks and object geometry are diagnostic only: isolate target
zoom and check surface error; they do not modify the fused scene cloud.

Save all per-frame fused XYZ/RGB/provenance in clouds.npz with ragged offsets,
RGB and provenance-colored binary PLY at frames 0/99/137, comparison.mp4 and
sample PNGs. Three cloud panels share the front camera's viewing direction;
point splats are a visualization, not a watertight mesh. Provenance PLY colors:
cyan=front only, orange=top only, purple=both views in the same 3 mm voxel.

Run `.venv/bin/python -u -m OATFlow.data_tools.fuse_episode` in tmux with tee.
Check source hashes, projection round trips, provenance count conservation,
cloud archive round trip and all video frames. config.json is the exact
protocol; records.json holds framewise counts. Results below.

Limitations: one simulated episode with ideal calibration and depth; no
real-sensor noise evaluation or measured policy gain. More observed surface
points do not imply more accurate object centers. All views are fused only
at the same timestamp to avoid smearing moving objects and robot links.


Results (complete):

| Quantity | Front | Top | Fused |
|---|---:|---:|---:|
| Mean scene points/frame | 65,774 | 62,040 | 87,862 |
| Initial red target points | 257 | 80 | 260 |
| GT object surface MAE | 0.559 mm | 0.450 mm | 0.576 mm |
| GT object surface P95 distance | 1.371 mm | 1.447 mm | 1.384 mm |

Raw single-view counts and voxel-downsampled fused counts have different
densities; point-count gain is not a surface-coverage metric. Averaged over
frames, 10.32% of top-view red-target points lie >3 mm from the front-view
red-target cloud. This is a sample-distance diagnostic, not exact new-area
coverage. No object-center estimator is fitted or evaluated.

NumPy concatenation/voxel aggregation takes 141.83 ms/frame on this machine,
excluding decode, backprojection, visualization and archive writes. Total
run time 74.14 s including video generation and compression. Full 138-frame
cloud archive: 74.78 MiB. Calibration is copied beside the cloud archive.

Projection round-trip maximum error <1e-8 pixels; known-coordinate geometry,
voxel centroid/RGB/provenance and box-distance checks pass. Source hashes
are unchanged, source counts conserve all input points, cloud archive and
binary PLY round trips are exact, and all 138 video frames decode.

Fusion adds observations without measurably improving surface error here;
the fused surface MAE is slightly higher due to sampling/voxel averaging.
It does not establish better object-center accuracy or policy success.
