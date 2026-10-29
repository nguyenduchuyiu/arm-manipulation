# From Reaching to Manipulation

The report is now organized as an English slide presentation, from basic reaching to end-to-end manipulation. Each slide has one main idea; implementation details and qualifications are in the speaker notes.

- [Slide-by-slide report and speaker notes](manipulation_slides/slides.md)
- [Editable PowerPoint](manipulation_slides/presentation.pptx)
- [Presentation PDF](manipulation_slides/presentation.pdf)
- [Slide overview](manipulation_slides/overview.png)
- [Detailed robot + translucent voxel workspace: web GLBs and viewer](manipulation_slides/assets3d/README.md)
- [Tracking video: GT and predicted object centers](manipulation_slides/videos/index.html)
- [Complete package: figures, code, data, and references](manipulation_slides/README.md)

The main presentation follows this sequence:

1. A 3D goal maps to an end-effector delta trajectory; IK computes joint commands.
2. Voxelization makes workspace coverage finite, followed by IK, collision-aware planning, and dynamics validation.
3. FM learns reaching trajectories and is evaluated on held-out goals.
4. A visual encoder and task ID condition FM for grasping, with auxiliary XYZ supervision shaping the latent.
5. Ablations test auxiliary supervision and initialization from weights learned on reaching.
6. An initial point prompt and TCOW provide object tracking through occlusion and shuffling.
7. TCOW and FM are jointly trained for cover removal, grasping, and lifting; results remain blank.

The earlier detailed record remains available as a [technical reference](top_pick_report_technical.md).
