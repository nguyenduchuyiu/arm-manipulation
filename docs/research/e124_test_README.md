# E124 segmentation test

Status: PLANNED.

Model: original frozen MobileSAM TinyViT/prompt encoder, learned four task tokens,
single-mask decoder; initialized E123 epoch5/step1280. Train prompt/decoder for
5additional epochs on all2048train episodes/296688frames, stride1, all4task
queries/frame, B64images/256queries, AdamW1e-3prompt/1e-4decoder, BCE+Dice+0.1quality.
Input RGB+target ID only. Original training/heldout partitions remain disjoint.

Select by best temporal validation IoU on original256validation episodes,
8uniform frames/episode, never by test. Exact selected checkpoint/step/hash and
case counts are in each test subdirectory's config.json and parent config.json.

Home:1408original heldout queries. Dynamic:72fixed later expert images x4queries.
XY:16retained paired inside/outside observations x4queries. The latter two are
diagnostics, not full-trajectory population estimates. Quality includes visible
mask IoU/Dice, task identity and confusion. Visible targets require8GTpixels;
low-visibility counts remain explicit. Prediction threshold0.5. GT masks are
labels only. Sample sheets show orange GT and cyan prediction for all4queries.

Segmentation only: no actions, IK, FM or manipulation rollout; H/K, flow steps,
noise seed and rollout frame budget are not applicable. No grasp success claim.
