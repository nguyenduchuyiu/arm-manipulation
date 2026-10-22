"""Causal point-query XYZ overlays on saved e96 top-camera clips."""
import json
from pathlib import Path
import subprocess

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from tqdm import tqdm

from OATFlow.tracking.point_data import point_clip, project, top_camera


@torch.no_grad()
def videos(model, episodes, root, output, step):
    description = ('# Point-query XYZ videos\n\nStatus: planned.\n'
        'Two saved-video cases: validation and both-unseen; trained model/settings in ../config.json. '
        '30past/current frames, initial dot only, top crop240x320,25/3FPS. '
        'GT/prediction and past trajectories are visualization only; no robot control.\n')
    if model.local_query:
        description += '\nFixed frame0 3x3-patch appearance reference.\n'
    dest = output / 'test_videos'; dest.mkdir(exist_ok=False)
    (dest / 'README.md').write_text(description)
    config = dict(checkpoint=str(output / 'best.pt'), step=step,
                  episodes=[ep['row']['path'] for ep in episodes], views=['top'],
                  fps=25 / 3, frame_stride=3, frames_per_prediction=30, query_frame=0,
                  inputs='raw top RGB prefix and radius3 dot at frame0 only',
                  local_appearance_reference=model.local_query,
                  mode='saved-video localization; no robot policy/actions/IK')
    (dest / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    camera = top_camera(root)
    model.eval()
    for case, ep in enumerate(episodes, 1):
        values = []
        for start in tqdm(range(0, len(ep['time']), 4), desc=f'Point video {case}', mininterval=5):
            clips = [point_clip(ep, end) for end in range(start, min(start + 4, len(ep['time'])))]
            rgb, query = (torch.cat([clip[j] for clip in clips]).cuda() for j in range(2))
            with torch.autocast('cuda', dtype=torch.bfloat16):
                values.extend(model.predict(rgb, query).cpu().numpy())
        prediction = np.array(values)
        measured_uv = project(ep['xyz'], camera); predicted_uv = project(prediction, camera)
        np.savez_compressed(dest / f'case_{case:02d}.npz', prediction_base_m=prediction,
                            truth_base_m=ep['xyz'], raw_index=ep['raw_index'], query_uv=ep['query_uv'])
        path = dest / f'case_{case:02d}.mp4'
        writer = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', '640x320', '-r', '25/3', '-i', '-', '-an',
            '-c:v', 'libx264', '-crf', '20', '-pix_fmt', 'yuv420p', str(path)], stdin=subprocess.PIPE)
        count = len(ep['context']['time'])
        for i in tqdm(range(len(prediction)), desc=f'Encode point video {case}', mininterval=5):
            source, index = (ep['context'], i) if i < count else (ep['expert'], i - count)
            image = source['rgb'][0, index].permute(1, 2, 0).numpy()
            canvas = Image.new('RGB', (640, 320), (18, 22, 30)); canvas.paste(Image.fromarray(image), (0, 60))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 6), f'{output.name} XYZ | {ep["row"]["path"]} | t={ep["raw_index"][i]/25:.2f}s', font=font, fill='white')
            draw.text((8, 28), 'Green=GT center  Magenta=prediction  Orange=initial query', font=font, fill='white')
            for uv, color in ((measured_uv, '#55ee88'), (predicted_uv, '#ee66dd')):
                points = uv[max(0, i - 25):i + 1] + [0, 60]
                if len(points) > 1:
                    draw.line([tuple(p) for p in points], fill=color, width=2)
                x, y = points[-1]; draw.ellipse((x-4, y-4, x+4, y+4), outline=color, width=2)
            if i == 0:
                x, y = ep['query_uv'] + [0, 60]
                draw.ellipse((x-3, y-3, x+3, y+3), fill='orange')
            error = float(np.linalg.norm(prediction[i] - ep['xyz'][i]) * 1000)
            draw.text((336, 70), 'Phase: ' + str(ep['phase'][i]), font=font, fill='white')
            draw.text((336, 100), f'XYZ error: {error:.1f} mm', font=font, fill='white')
            draw.text((336, 136), 'GT base XYZ (m)', font=font, fill='#55ee88')
            draw.text((336, 158), ' '.join(f'{v:.3f}' for v in ep['xyz'][i]), font=font, fill='white')
            draw.text((336, 200), 'Predicted base XYZ (m)', font=font, fill='#ee66dd')
            draw.text((336, 222), ' '.join(f'{v:.3f}' for v in prediction[i]), font=font, fill='white')
            writer.stdin.write(np.asarray(canvas).tobytes())
            if i == 0:
                canvas.save(dest / f'case_{case:02d}.png')
        writer.stdin.close(); assert writer.wait() == 0
        with av.open(str(path)) as video:
            assert sum(1 for _ in video.decode(video=0)) == len(prediction)
    with (dest / 'README.md').open('a') as stream:
        stream.write('\nComplete: both videos fully decoded; exact settings in config.json.\n')
    text = (dest / 'README.md').read_text().replace(
        'Status: planned.',
        'Status: complete; checkpoint and best step in config.json.')
    (dest / 'README.md').write_text(text)
