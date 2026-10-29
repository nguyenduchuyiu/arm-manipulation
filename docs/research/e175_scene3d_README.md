# E175 — Four task predictions in the same RGB scene, 3D rendering

Status: complete.

User asked all objects in one scene with predictions on the RGB scene, explicitly3Drender rather than2Dprojection. Use exact E157IIDscene00000, one full4096point input/SAM3features/pixelgraph, infer all4taskIDs together with the same frozen best12000snapshot used in test_best.4×128rawpredictedXYZ are displayed in their originalrobotbaseframe with red/blue/green/yellow taskcolors. No GTcentroid/size/rotation alignment. Sourcecamera/depth/RGB unchanged. E17550ktraining continues independently.

Render: textured RGB-D triangle surface in3D, vertexpositions backprojected via exactintrinsics and camera_to_base. Imagegrid stride2, omit invaliddepth and triangles acrossdepthjumps; this represents the visible observation, not unknown backfaces of simulatorassets. Texture originalRGB. Three.js0.160.1/OrbitControls vendored with MITlicense and contenthashes, allassets served locally from existingforwarded server. Initialcamera matches originalpose andverticalFOV; buttons originalcamera,fullscene,focus4objects; orbit/zoom/pan, per-task visibility and pointsize. DefaultXraypointdisplay makes every predictionvisible while preserving3Dcoordinates; canenabledepthtest. No flat2Dpointoverlay is generated. SidecarRGBPNG is only the meshtexture.

Inference only,772018encoder/decoderparams frozen for this test, FP32/TF32off, one scene/4tasks/128points. No training/learningrate/noise/flowsteps; originalsourceE175training wascanonical/noSE3/50kEMD. Check checkpoint/modelhash,taskcounts,finiteXYZ,drift against same saved scene predictions, and calibrationprojection roundtrip to RGBpixels. Save predictionNPZ,mesh buffers, local3Dviewer and metadata. Checkpoint referenced from immutable test_best snapshot to avoid duplicateweights.

Own diag/e175-scene3dworktree. Selective model/projection helper copies plus hashes; vendorURLs/hashes in vendor_sources.json. No imports from activeexperiment code. Run `/workspace/arm-manipulation/.venv-concerto/bin/python code/run.py` from testroot; creates freshresults. Attachedtmux ssh_tmux:e175-scene,stdout/stderr mirrored to run.log. Viewer existingforwarded localhost:17070/e175_scene3d/ . Screenshots/browserverification appended afterrender. Canonical notes docs/research/e175_scene3d.md.

Checkpointstep12000, codecommit `0692a112a20134bde584829fbf6aa6cbfad65011`. See results/summary.json.

Browser verification passed in actualChromium/WebGL: localassetsloaded,no pageerrors; nativecamera projection matches calibration within0.01872pixel; toggles/presets work; native/focus/overview screenshots inspected. Runtimeassets ~1546452bytes, allservedlocally. Scene3Dcontains18588vertices/35337faces and4×128predictedpoints from the same IIDscene00000,best12000. Calibration roundtrip1.086e-5pixel; freshinference drift0.000152mm vs same-snapshot savedscene. SourceRGB/depth/predXYZ unmodified; unknownhidden surfaces not reconstructed.

Viewer: http://localhost:17070/e175_scene3d/
