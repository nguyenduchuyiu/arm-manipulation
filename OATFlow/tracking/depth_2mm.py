"""Compress the existing exact depth reference at2mm and preview the decoded file."""
import json
from pathlib import Path
import shutil
import subprocess
import time
import zlib

import numpy as np
from tqdm import tqdm

from OATFlow.tracking.depth_compress import geometry_errors
from OATFlow.tracking.pointcloud_camera_video import main as visualize
from OATFlow.tracking.mask_video import digest


SOURCE = Path('/workspace/robotics_assets/experiments/e114_mask/test_depth_compress')
OUT = SOURCE.parent/'test_depth_compress_2'
STEP = .002
README = Path('docs/research/e114_2mm_README.md')


def main():
    OUT.mkdir(exist_ok=False)
    shutil.copyfile(README, OUT/'README.md'); shutil.copyfile(__file__, OUT/'source.py')
    reference = SOURCE/'depth_float32.zlib'; reference_hash = digest(reference)
    previous = json.loads((SOURCE/'config.json').read_text())
    assert previous['status'] == 'complete'
    original = np.frombuffer(zlib.decompress(reference.read_bytes()), dtype='<f4').reshape(previous['frames'], 240, 320)
    valid = (original > 0) & (original <= 65535*.0001)
    quantized = np.zeros(original.shape, dtype='<u2')
    quantized[valid] = np.rint(original[valid]/STEP).astype('uint16')
    config = dict(status='running', case=previous['case'], source_camera=previous['source_camera'],
        frames=previous['frames'], fps=25, resolution=[320, 240], unit_m=STEP,
        reference=str(reference), reference_sha256=reference_hash, common_valid_range_m=[0, 6.5535],
        invalid='same pixels as original benchmark; code0 invalid',
        codec='FFV1 gray16le level3 coder1 context1 g1 slices4 slicecrc1 threads4', training='none')
    (OUT/'config.json').write_text(json.dumps(config, indent=2))
    path = OUT/'depth_uint16_2mm.mkv'; start = time.perf_counter()
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'gray16le', '-s', '320x240', '-r', '25', '-i', '-', '-an', '-c:v', 'ffv1',
        '-level', '3', '-coder', '1', '-context', '1', '-g', '1', '-slices', '4', '-slicecrc', '1',
        '-threads', '4', str(path)], stdin=subprocess.PIPE)
    for frame in tqdm(quantized, desc='Encode FFV1 2mm', mininterval=5): encoder.stdin.write(frame.tobytes())
    encoder.stdin.close(); assert encoder.wait() == 0
    encode_s = time.perf_counter()-start
    start = time.perf_counter()
    decoded = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-threads', '4',
        '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'gray16le', '-'], capture_output=True, check=True).stdout
    restored = np.frombuffer(decoded, dtype='<u2').reshape(original.shape)
    assert np.array_equal(restored, quantized)
    decode_s = time.perf_counter()-start
    camera = dict(focal=160/np.tan(np.deg2rad(previous['source_camera']['fovy_deg'])/2))
    errors = geometry_errors(original, restored.astype('float64')*STEP, valid, camera)
    baselines = json.loads((SOURCE/'results.json').read_text())
    base_sizes = {r['step_m']: r['bytes'] for r in baselines if 'step_m' in r}
    result = dict(format='uint16 2mm + FFV1', bytes=path.stat().st_size, raw_bytes=quantized.nbytes,
        ratio_vs_native_raw=quantized.nbytes/path.stat().st_size, codec_max_error_units=0,
        invalid_pixels=int((~valid).sum()), errors_against_float32=errors, encode_s=encode_s, decode_s=decode_s,
        reduction_vs_1mm=1-path.stat().st_size/base_sizes[.001],
        reduction_vs_0p1mm=1-path.stat().st_size/base_sizes[.0001],
        gb_per_hour=path.stat().st_size/(len(original)/25)*3600/1e9)
    assert digest(reference) == reference_hash
    (OUT/'results.json').write_text(json.dumps(result, indent=2))
    config['status'] = 'complete'; (OUT/'config.json').write_text(json.dumps(config, indent=2))
    print('COMPRESSION', json.dumps(result), flush=True)
    del original, quantized, restored, decoded
    visualize(depth_file=path, depth_unit_m=STEP, out=OUT/'view_uint16', readme=README)


if __name__ == '__main__':
    main()
