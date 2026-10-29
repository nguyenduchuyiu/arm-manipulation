# From Reaching to Manipulation — presentation package

Start with [presentation.pdf](presentation.pdf), or open [presentation.pptx](presentation.pptx) to edit the slides. [overview.png](overview.png) shows the complete deck at a glance.

The presentation is in English: **19 main slides + 3 appendix slides**, with one main idea per slide. Detailed numbers, caveats, and implementation notes are in the speaker notes rather than crowded onto the slides.

The narrative follows the project idea:

1. Start with the basic motion interface: 3D goal → FM → end-effector deltas → IK → joint commands.
2. Discretize the continuous workspace into 2 cm voxels and validate candidate goals physically.
3. Train FM on reaching trajectories and evaluate held-out reaching targets.
4. Add observation/task conditioning and behavior cloning to learn grasping; use auxiliary XYZ supervision to shape the visual latent.
5. Show the auxiliary-loss and FM-weight-initialization ablations.
6. Introduce an initial point prompt and TCOW for tracking through occlusion/shuffling, with a separately labeled qualitative tracking-pilot video.
7. Combine TCOW and FM end-to-end. The FM uses weights learned on the reach task. Final results remain blank.

## Files

| File or folder | Purpose |
|---|---|
| `presentation.pptx` | Editable slide titles, body text, tables, and embedded speaker notes. Figures are high-resolution images; their vector sources are supplied separately. |
| `presentation.pdf` | Matching PDF for reading and presenting. |
| `slides.md` | Slide-by-slide report with the main message and speaker notes. |
| `slides.json` | Structured slide content for further editing or conversion. |
| `overview.png` | Contact sheet of all 22 slides. |
| `slides/` | Individual slide PNG previews. |
| `figures/` | 16 reusable figures, each as SVG, PDF, and 240 dpi PNG. |
| `figure_captions.md` | Captions, interpretation, and figure provenance for paper reuse. |
| `data/` | Small portable data snapshots, source images, metrics, and CSV tables. No datasets or checkpoints need to be downloaded to rebuild the figures. |
| `code/` | Python source for input extraction, vector figures, slide content, and deck generation. |
| `references/` | Source notes, TCOW citation, architecture Mermaid source, and design guidelines. |
| `videos/` | GT/prediction tracking MP4s, posters, offline player, frozen sources, and provenance. The PPTX embeds the combined demo; PDF contains its poster. |
| `assets3d/` | Importable GLBs of the detailed arm and translucent voxel workspace, an offline web viewer, and rendered previews. See its README for import instructions. |

## Rebuild

Use Python 3.12 with the packages in `code/requirements.txt`. From this package folder:

```bash
python code/figures.py
python code/build.py
```

The frozen inputs in `data/` make these two commands portable and CPU-only. They do not load a learned model or use CUDA.

`code/prepare_inputs.py` is an optional extraction step for the original project environment. It requires the original `/workspace/robotics_assets` tree, MuJoCo, PyAV, and the project source. The frozen data already provided are sufficient for normal rebuilding.

## Figure use

Use **PDF or SVG** for a paper; SVG retains text labels and PDF embeds TrueType fonts. PNG is convenient for slides. The workspace uses actual validated voxel centers and the detailed original robot visual mesh at its recorded home pose, rendered from the supplied GLB scene. Its SVG/PDF embeds that raster render with vector labels. The trajectory plot uses a real validated expert demonstration and a simplified robot. Scene panels are real recorded observations, not model predictions. The flow-matching interpolation is an explicitly conceptual illustration, not a measured rollout.

All architecture blocks are labeled **FM**. Its initialization from weights trained on reaching is explained in the narrative and speaker notes. TCOW is initialized from its original official checkpoint, not from the earlier tracker pilots.

Results for e105 are deliberately empty in the PPTX, PDF, Markdown, JSON, and CSV. No training loss or old tracking-pilot result is substituted for end-to-end manipulation performance.
