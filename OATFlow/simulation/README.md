# Shared robot execution and rendering

pose.py: EE labels,rotation6D,absolute-jaw transform and bounded5joint IK.
control.py:25Hz action,50Hz midpoint servo,500Hz physics.
render.py: original user meshes/lighting and trajectory markers.
io.py: JSON and FFmpeg RGB encoder. contacts.py: physical bilateral/lift checks.
All robot references/meshes/home are in robotics_assets/robot/nexarm.
