"""Small JSON and RGB-video output helpers."""
import json
import subprocess

def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def encoder(path,width,height):
    return subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s',f'{width}x{height}','-r','25','-i','pipe:0','-an','-c:v','libx264','-threads','2',
        '-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
