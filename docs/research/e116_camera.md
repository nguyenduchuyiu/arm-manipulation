# E116 wrist camera assumptions

User hardware: fisheye camera, exact model unknown. Full physics collection continues. The wrist configuration is now resolved:
front RGB-D plus synthetic fisheye wrist RGB, no wrist depth. The temporary render
hold was released after camera/replay checks. No model training.

Manufacturer specifications and manipulation reference checked 2026-10-20:

| Reference | Image/depth | Relevant limitation |
| --- | --- | --- |
| Arducam B0261 IMX291 | RGB, 175-degree diagonal fisheye, USB UVC, M12 manual focus | No depth stream; close-focus distance needs measurement with the actual lens setting. |
| GoPro HERO9 in UMI | Wrist fisheye RGB | UMI uses a wrist GoPro; this does not supply a depth camera stream. |
| RealSense D405 | Stereo RGB-D, 87 x 58-degree FOV | Standard min-Z 7cm at480p; ideal range7-50cm. Not a fisheye match. |
| Luxonis OAK-D Pro W | Wide stereo RGB-D; OV9782 variant150-degree diagonal | Min-Z about20cm extended/400p,37cm800p; ideal40cm-6m. Poor fit for our final8-9cm lens-to-object distances without changing mounting. |

Sources:
- [Arducam manufacturer product page](https://www.arducam.com/arducam-fisheye-low-light-usb-camera-for-computer-2mp-1080p-imx291-wide-angle-mini-h-264-uvc-video-camera-board-with-microphone.html)
- [Arducam selection guide (manual focus)](https://docs.arducam.com/UVC-Camera/USB2-UVC-Camera-Kit/Specs-and-Selection-Guide/)
- [UMI author appendix: wrist HERO9 setup](https://umi-gripper.github.io/umi.pdf)
- [RealSense D405 manufacturer specifications](https://www.realsenseai.com/product-family/d405-series/)
- [OAK-D Pro W manufacturer specifications](https://checkout.luxonis.com/products/oak-d-pro-w)

Recommendation depends on the observation available at deployment. For existing
single-lens fisheye RGB hardware, emulate fisheye projection and calibrate the
actual lens/mount; simulated wrist depth is GT, not an available real input. For
physical wrist RGB-D at short distances, D405 is a relevant reference, but its
perspective FOV and minimum range differ from a USB fisheye. These are reference
options, not an identification of the user's camera or a popularity ranking.

Keep geometric clipping separate from sensor range. MuJoCo's original near plane
was8cm due to scene extent8m; clipping removed near occluders and let the camera
see the target through them. Absolute1mm/10m planes correct geometric rendering.
A physical depth minimum should invalidate the measured foreground depth without
revealing surfaces behind it. RGB can remain present, possibly defocused or
occluded; it must not be hard-blanked solely by a depth minimum. Focus cannot be
inferred from FOV alone. A fisheye projection also needs its distortion/ray model;
simply increasing a pinhole FOV does not reproduce it.

Evidence: e116_dynamic/preview/wrist_clipping.png and wrist_clip_validation.json.
The four-frame validation passes RGB decode, exact FFV1 integer-depth roundtrip,
calibration fields, and2mm depth quantization. Initial full pilot videos predate
the clip fix. No calibrated physical sensor model or corrected fisheye dataset
has been produced yet.

## Selected simulation configuration

The user dropped wrist depth and asked to rotate the wrist preview180deg. Use the
B0261 nominal175deg diagonal FOV as a reference for synthetic equidistant RGB.
This choice does not identify or calibrate the user's real hardware. Keep front
RGB-D and both cameras' masks/calibration. No wrist depth stream or depth input.
Fisheye RGB and labels use the same five-face projection; ideal fisheye K and four
zero distortion coefficients are stored explicitly. Raw sensor orientation is
preserved; only the preview rotates wrist RGB and labels180deg.

A scene-ray check at576 pixels over three frames agrees with576 simulated rays on
575 semantic IDs; one boundary pixel differs due to rasterization. No focus model
or arbitrary minimum RGB distance is claimed. The initial pinhole wrist XML is the
physical mounting reference only; OATFlow/data_tools/wrist_camera.py defines the
image projection used by the current renderer.

Final verification: current-camera preflight2contexts/8expert queries PASSED;
10segments3391frames, full replay error0 and front depth codec bit-exact. Both
upright preview clips fully decoded554/678frames. Finished preflight payloads
removed after retaining JSON summaries, manifest and sample calibration. Full
render hold released; collection pipeline continues automatically.
