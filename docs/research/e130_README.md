# E130 — probes of the existing E122 geometry decoder

Status: COMPLETE. Architecture and checkpoint remain unchanged; no training.

Checkpoint: e122_full/train/final.pt, step 23520. Original training: full
2048 E117 pick demonstrations, 32 layouts, stride 1, batch 128, 23520 updates
(10.217 full-data epochs), frozen ImageNet ViT-B/16, FM initialized from E118
point-cloud reach step120000. Original optimizer/LR/normalization are copied
from the checkpoint into config.json; no optimizer is constructed here.

Reuse all 8 matched x-boundary pairs (4 colors x 2 edges) from E122 test_xy:
inside 30 mm and outside 30 mm relative to pick-training X[0.30,0.78] m,
Y[-0.42,-0.10] m. Re-render saved initial states and pre-action replans at
0,75,150,225,285 when present in the original traces. No new simulation
steps, expert actions, FM action sampling or rollout outcomes. Historical
traces used CPU FP32 H25/K15/Euler10/seed0/300frames. Probe inference is CUDA
FP32 with TF32 disabled. H/K/flow steps/noise do not apply to the deterministic
auxiliary cloud head itself. The near-boundary inside controls were weak
in the historical rollout; this is a diagnostic pilot, not a population test.

1. Read-only hooks capture both decoder cross-attention layers (4 heads each).
Reconstruct softmax QK weights and verify their value/output projection against
the actual unchanged attention output. Report target patch mass, uniform-prior
lift, pointing hits and object ranking, alongside predicted-vs-measured target
surface Chamfer and centroid errors. GT masks only score/visualize attention;
they are not model inputs. Attention alone does not establish causal selection.

2. Compare GT-mask-weighted RGB token features of the same object in each
physical pair, with different-object negatives. Also roll exactly the same
pre-position patch tokens horizontally by +/-1,2,4 columns. Compare original
fixed positional embeddings versus positional embeddings co-moved with the
tokens. The latter is an input-level equivariance control using existing
encoder modules, not a weight or architecture change. Circular token rolls
are synthetic controls; physical pairs additionally change viewpoint/context.

3. Keep exact cached RGB features/target fixed. Change only camera_to_base
translation at the existing depth/calibration interface: +/-5,10,20,40,60,100 mm
on each axis. Check back-projected XYZ changes by the prescribed translation.
Measure centroid gain, off-axis response, deformation and translation-compensated
Chamfer to the baseline prediction + translation. This shifts all valid scene
points while pixels/joints remain fixed: a counterfactual geometry-sensitivity
probe, not a physically consistent camera/robot or an FM rollout.

Run `MUJOCO_GL=egl .venv/bin/python -u -m OATFlow.top_pick.geometry_probe`
in tmux with tee. Output: /workspace/robotics_assets/experiments/e130_probe.
Save exact config, records, attention/feature/cloud arrays and figures. Verify
checkpoint/source/model state hashes, hook parity, zero-shift identity,
position co-movement control and prescribed XYZ changes. Results below.


## Results

Completed 75 saved snapshots (35 inside, 40 outside), including 16 paired
initial scenes; 96 controlled RGB token rolls and 576 fixed-feature XYZ
translation trials. No architecture, checkpoint or model-state changes.

### 1. Semantic attention versus cloud reconstruction

| Initial scene group | Layer-1 peak hits target patch | Layer-1 object rank correct | Mean surface Chamfer | Mean surface-centroid error |
|---|---:|---:|---:|---:|
| Inside, 8 scenes | 8/8 | 8/8 | 3.058 mm | 6.512 mm |
| Outside, 8 scenes | 8/8 | 8/8 | 13.554 mm | 25.783 mm |

The fourth head of layer 1 puts approximately 99% of its mass on patches
intersecting the target in both groups. Other heads attend to context, so
the four-head average target mass is approximately 25%, not 99%. Layer 2
is predominantly contextual: its mean-attention peak hits the target in
0/8 initial scenes in both groups; it also receives the residual query
from layer 1. Do not describe every attention layer as target-localized.

All 8 outside predicted cloud centroids are displaced inward along X
relative to the measured visible target-surface centroid: mean 25.035 mm.
Across physical pairs, cloud-centroid X displacement is a median 0.625 of
the measured visible-surface centroid displacement. This supports regression
contraction after a target-selective attention stage. It does not isolate
the failure to the final linear cloud head: geometry embedding, fusion,
decoder, position embeddings and cloud head remain possible contributors.

Later outside snapshots: layer-1 object ranking is correct 32/32, while
its mean-attention peak touches target patches in 20/32. Outside mean
surface Chamfer is 39.42 mm versus 25.99 mm in the 27 inside later samples.
These later samples differ in trajectories/occlusion and counts; they are
not matched causal evidence for the extrapolation effect.

### 2. RGB feature stability

GT-mask-fraction-weighted final ViT patch pooling gives mean same-object
inside/outside cosine 0.8958, versus 0.5758 for different objects in the
same images. Physical pairs include changes to perspective, background
within boundary patches and lighting. Pre-position patch pooling cosine
0.4871 is sensitive to these pixel/patch effects; it is not semantic accuracy.

For exactly identical patch content rolled to new token positions, final
feature cosine averages 0.9812 with original fixed positions: approximately
0.995 at +/-16px, 0.988 at +/-32px and 0.960 at +/-64px. Co-moving positions
with content restores cosine to approximately 1.000000; the minimum is
0.99999988. This isolates an absolute-position effect in the frozen ViT,
but feature similarity alone does not quantify its contribution to cloud
error or establish it as the dominant cause. Circular token rolls are
synthetic controls, not physically rendered camera motion.

### 3. Fixed RGB feature, translated XYZ input

Only the existing camera_to_base translation changes. Back-projection
changes every valid XYZ by the requested offset to within 0.000142 mm.
The same cached RGB features and target ID are reused throughout.

| Input change | Median output centroid dX | Median dY | Median dZ |
|---|---:|---:|---:|
| +10 mm X | +5.232 mm | +0.843 mm | +0.035 mm |
| +10 mm Y | +0.337 mm | +7.992 mm | -0.016 mm |
| +10 mm Z | +0.054 mm | -9.138 mm | +0.021 mm |

Over +/-5,10,20mm trials, median same-axis gain is X=0.491, Y=0.774,
Z=0.00258; ideal translation-equivariant reconstruction would have gain 1
and zero off-axis response. A separate central-difference estimate has
strong Z-to-Y coupling around -0.902. These numbers quantify response to modified extrinsics, but do not establish
a clean physical translation-equivariance failure or a particular internal
cause. The shift moves the full geometry while keeping RGB/joints fixed,
so the inputs no longer form a consistent physical scene or coordinate-frame
change. This is a counterfactual sensitivity test, not an FM control or
real-camera robustness result.

### Interpretation and artifacts

The strongest evidence is target-selective first-layer attention combined
with contracted/non-equivariant coordinate reconstruction. RGB absolute
positions have a measured effect, but the current probes do not establish
them as the primary cause. No architecture change is made or recommended
as an experimentally proven fix by this test.

Figures: target_attention_x_low.png, target_attention_x_high.png (layer-1
inside/outside); attention_x_low.png, attention_x_high.png (both layers);
attention_history_*.png (outside initial/later); cloud_xy.png; rgb_features.png;
xyz_response.png; geometry_bias.png. All raw attention heads, clouds, masks
and features plus per-trial records are retained for audit.

Source/checkpoint/model-state hashes are unchanged. Read-only hooks give
bit-exact cloud output in the parity check; reconstructed attention output
maximum discrepancy is 4.30e-6. Zero-shift cloud discrepancy is 8.95e-8 m
from batch-size numerics. All checks pass in verification.json.

Reproduce inference with geometry_probe above using a fresh output path;
` .venv/bin/python -m OATFlow.top_pick.probe_analysis` summarizes saved arrays
and adds geometry_bias.png without model inference. The analysis source
hash is separate from the original inference-source hash in config.json.


Follow-up: the output-only oracle centroid test in e130_center_README.md
reduces initial OOD surface Chamfer from 13.554 to 3.600 mm, compared with
3.382 mm for centroid-aligned ID. It provides direct evidence that the large
OOD gap is dominated by placement for these fixed-size cube surfaces.
