# E171 — Centroid alignment at last18000

Status: complete.

Diagnostic only, no training or GPU inference. Use saved E171last18000 predictions from exactly the original20Eulersteps/noise167. For each query shift all128predictedpoints by GT128centroid minus predictedcentroid. No rotation, scale or point matching is applied to the output. GT centroid is oracle supervision used for diagnosis; this is not GT-free inference or a deployable correction.

Checkpoint E171train/last.pt step18000, verified against saved summary hash. Frozen E168Z+fullmemory, FM/adapters580483params; original training18000steps,1024scenes,4tasks,B8scenes,FP32,HungarianvelocityMSE,noSE3,AdamW3e-4→3e-5. No weights changed here, no learning rate/epochs for this test. Retain completed E171 and active E172 code/checkpoints untouched; own branch/worktree diag/e171-align18k.

Test all128IIDscenes×4tasks, plus the same4cachedSE3views (512canonical+2048SE3queries),128predicted/GTpoints. Reconstruct visible fullGTsurface from immutableE157depth/labels for voxel4mm coverage@2/5mm. Report exactHungarianEMD,directionalChamfer,CD,centroiderror andsurfacecoverage before/after. Assert centeredpointgeometry and XYZ extents unchanged, centroids match, rawEMD/alignedCD reproduce prior evaluation. Visualize fixed scenes0/1/2/3,tasks0/1/2/3 on IID andSE3view1; buttons toggle aligned/raw. One canonical staticPNG for visual inspection. No new frame generation or random sampling.

Own selective coverage/projection helper copies, original hashes in sources.json. Commit recorded with artifacts. Run `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py` from test root. Creates fresh results directory. stdout+stderr in attachedtmux ssh_tmux:e171-align and run.log. Viewer on existing forwarded localhost:17070/e171_align18k.html. Canonical notes docs/research/e171_align18k.md.

Code commit `2cfc0bc795efe8456944c3be422e16921f9b34aa`. See results/summary.json and results/RESULTS.md.

# E17118k centroid alignment diagnostic

p_aligned=p+mean(GT128)-mean(p); translation only, oracle GT centroid, no rotation/scale or new inference

| Test/method | EMD mm | CD mm | Centroid mm | Coverage5mm % |
|---|---:|---:|---:|---:|
| IID/raw | 11.266 | 6.719 | 8.382 | 45.16 |
| IID/aligned | 7.499 | 5.066 | 0.000 | 61.12 |
| SE3/raw | 312.990 | 291.878 | 312.437 | 1.57 |
| SE3/aligned | 13.857 | 9.150 | 0.000 | 31.10 |

Oracle translation diagnostic, not deployable inference. Shape/size/orientation unchanged. Same stored128points/noise/20Eulersteps.
Code commit `2cfc0bc795efe8456944c3be422e16921f9b34aa`.

Viewer localhost:17070/e171_align18k.html HTTP200,4939230bytes; defaults to aligned, raw toggle. Static IID preview inspected. No source model/training/prediction modified; E17250k continuation still running independently.
