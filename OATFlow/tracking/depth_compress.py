"""Measure actual depth-video storage and separate quantization from codec error."""
from io import BytesIO
import json
from pathlib import Path
import shutil
import subprocess
import time
import zipfile
import zlib

import mujoco
import numpy as np
from PIL import Image
from tqdm import tqdm

from OATFlow.data_tools.collect_shuffle import NAMES
from OATFlow.simulation.render import visual_model
from OATFlow.tracking.mask_video import DATA
from OATFlow.tracking.point_data import top_camera
from OATFlow.tracking.pointcloud_front import configure


OUT = Path('/workspace/robotics_assets/experiments/e114_mask/test_depth_compress')
REF = Path('/workspace/robotics_assets/experiments/e114_mask/test_pointcloud_3')


def geometry_errors(original, decoded, valid, camera):
    error = (decoded.astype('float64')-original)[valid]*1000
    yy, xx = np.indices(original.shape[1:])
    ray_length = np.sqrt(1+((xx-159.5)/camera['focal'])**2+((yy+40-159.5)/camera['focal'])**2)
    xyz_error = (np.abs(decoded-original)*ray_length)[valid]*1000
    return dict(valid_pixels=int(valid.sum()), depth_mae_mm=float(np.abs(error).mean()),
                depth_rmse_mm=float(np.sqrt(np.mean(error**2))), depth_max_mm=float(np.abs(error).max()),
                depth_p95_mm=float(np.quantile(np.abs(error), .95)),
                xyz_mae_mm=float(xyz_error.mean()), xyz_max_mm=float(xyz_error.max()))


def main():
    OUT.mkdir(exist_ok=False)
    shutil.copyfile('docs/research/e114_depth_compress_README.md', OUT/'README.md')
    shutil.copyfile(__file__, OUT/'source.py')
    config = json.loads((REF/'config.json').read_text())
    case = config['cases'][0]; row = case['row']
    qpos = np.concatenate([np.load(DATA/ref/'sim_state.npz')['qpos'] for ref in (row['context_path'], row['path'])])
    settings = dict(status='running', case=case, source_camera=config['source_camera'],
                    fps=25, resolution=[320, 240], crop=[40, 280], frames=len(qpos),
                    signal='native axial camera depth in metres, ideal MuJoCo renderer',
                    float32_codec='zlib level6, exact whole-video byte stream',
                    uint16_steps_m=[.0001, .001], uint16_codec='FFV1 level3 gray16le; coder1 context1 slices4 slicecrc1',
                    png='16-bit PNG compress_level6 in ZIP_STORED, step0.1mm',
                    invalid='uint16 code0 for nonfinite/nonpositive/out-of-range depth; mask is explicit',
                    errors='codec error against encoded representation; geometry against float32 over common valid mask only',
                    training='none; no policy/tracker inference')
    (OUT/'config.json').write_text(json.dumps(settings, indent=2))
    model = visual_model(DATA/'scene.xml', object_bodies=NAMES)
    model.vis.global_.offwidth = model.vis.global_.offheight = 320
    state = mujoco.MjData(model)
    camera = configure(model, state, top_camera(DATA)['base'])
    option = mujoco.MjvOption(); option.geomgroup[3] = 0
    original = np.empty((len(qpos), 240, 320), dtype='<f4')
    with mujoco.Renderer(model, height=320, width=320) as renderer:
        renderer.enable_depth_rendering()
        for i in tqdm(range(len(qpos)), desc='Render original depth', mininterval=5):
            state.qpos[:] = qpos[i]; mujoco.mj_forward(model, state)
            renderer.update_scene(state, camera='cam_left', scene_option=option)
            original[i] = renderer.render()[40:280]
    assert np.isfinite(original).all() and (original > 0).all()
    results = []
    start = time.perf_counter()
    packed = zlib.compress(original.tobytes(), level=6)
    path = OUT/'depth_float32.zlib'; path.write_bytes(packed)
    encode_s = time.perf_counter()-start
    start = time.perf_counter()
    restored = np.frombuffer(zlib.decompress(packed), dtype='<f4').reshape(original.shape)
    assert np.array_equal(restored.view('uint32'), original.view('uint32'))
    results.append(dict(format='float32 + zlib', bytes=path.stat().st_size,
                        ratio_vs_native_raw=original.nbytes/path.stat().st_size,
                        codec_max_error=0, geometry_max_error_mm=0, all_pixels_preserved=True,
                        encode_s=encode_s, decode_s=time.perf_counter()-start))
    del packed, restored
    # A real depth sensor has a finite range; use one common representable range.
    valid = (original > 0) & (original <= np.iinfo('uint16').max*.0001)
    assert valid.any()
    for step in (.0001, .001):
        quantized = np.zeros(original.shape, dtype='<u2')
        quantized[valid] = np.rint(original[valid]/step).astype('uint16')
        label = '0p1mm' if step == .0001 else '1mm'
        path = OUT/f'depth_uint16_{label}.mkv'
        start = time.perf_counter()
        encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-s', '320x240', '-r', '25', '-i', '-', '-an',
            '-c:v', 'ffv1', '-level', '3', '-coder', '1', '-context', '1', '-g', '1',
            '-slices', '4', '-slicecrc', '1', '-threads', '4', str(path)], stdin=subprocess.PIPE)
        for frame in tqdm(quantized, desc=f'Encode FFV1 {label}', mininterval=5):
            encoder.stdin.write(frame.tobytes())
        encoder.stdin.close(); assert encoder.wait() == 0
        encode_s = time.perf_counter()-start
        start = time.perf_counter()
        decoded = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '4',
            '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'], capture_output=True, check=True).stdout
        restored = np.frombuffer(decoded, dtype='<u2').reshape(quantized.shape)
        assert np.array_equal(restored, quantized)
        decode_s = time.perf_counter()-start
        errors = geometry_errors(original, restored.astype('float64')*step, valid, camera)
        record = dict(format=f'uint16 {label} + FFV1', step_m=step, bytes=path.stat().st_size,
                      raw_bytes=quantized.nbytes, ratio_vs_native_raw=quantized.nbytes/path.stat().st_size,
                      codec_max_error_units=0, invalid_pixels=int((~valid).sum()),
                      errors_against_float32=errors, encode_s=encode_s, decode_s=decode_s)
        results.append(record)
        del decoded, restored
        if step == .0001:
            path = OUT/'depth_uint16_0p1mm_png.zip'
            start = time.perf_counter()
            with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
                for i, frame in enumerate(tqdm(quantized, desc='Encode PNG16 sequence', mininterval=5)):
                    buffer = BytesIO(); Image.fromarray(frame).save(buffer, format='PNG', compress_level=6)
                    archive.writestr(f'{i:06d}.png', buffer.getvalue())
            png_encode_s = time.perf_counter()-start
            start = time.perf_counter()
            with zipfile.ZipFile(path) as archive:
                assert len(archive.namelist()) == len(quantized)
                for i in tqdm(range(len(quantized)), desc='Decode / verify PNG16', mininterval=5):
                    restored = np.asarray(Image.open(BytesIO(archive.read(f'{i:06d}.png'))))
                    assert np.array_equal(restored, quantized[i])
            results.append(dict(format='uint16 0p1mm + PNG16 ZIP', bytes=path.stat().st_size,
                raw_bytes=quantized.nbytes, ratio_vs_native_raw=quantized.nbytes/path.stat().st_size,
                codec_max_error_units=0, errors_against_float32=errors,
                encode_s=png_encode_s, decode_s=time.perf_counter()-start))
        (OUT/'results.json').write_text(json.dumps(results, indent=2))
        print('RESULT', json.dumps(record), flush=True)
    settings.update(status='complete', raw_float32_bytes=original.nbytes,
                    raw_uint16_bytes=original.size*2, valid_fraction=float(valid.mean()),
                    invalid_depth_range_m=[float(original[~valid].min()), float(original[~valid].max())] if (~valid).any() else None)
    (OUT/'results.json').write_text(json.dumps(results, indent=2))
    (OUT/'config.json').write_text(json.dumps(settings, indent=2))
    print('COMPLETE', json.dumps(dict(config=settings, results=results)), flush=True)


if __name__ == '__main__':
    main()
