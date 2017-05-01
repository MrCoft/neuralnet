import os
import subprocess
import numpy as np

FFMPEG_SCALE = "-vf scale=w={}:h={}:force_original_aspect_ratio=decrease"

def fix_avi(source, target):
    os.system("ffmpeg -i {} -vcodec mpeg2video -qscale 1 -qmin 1 -intra -vtag M701 {}".format(source, target))

def get_frame(path, time, size):
    width, height = size
    cmd = "ffmpeg -ss {} -i {} -vframes 1 -s {}x{} -f image2pipe -pix_fmt rgb24 -vcodec rawvideo -".format(time, path, width, height)
    data = subprocess.check_output(cmd, shell=True)
    img = np.fromstring(data, "uint8").reshape(height, width, 3)
    img = np.transpose(img, (1, 0, 2))
    return img

def get_info(path):
    cmd = "ffmpeg -i {}".format(path)
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, shell=True)
    out, err = proc.communicate()
    output = err.decode("utf-8")
    info = {}
    for line in output.splitlines():
        if "Duration:" in line:
            t = line.split()[1]
            h, m, s = t.split(':')
            info["duration"] = int(h) * 3600 + int(m) * 60 + float(s[:-1])
        if "fps," in line:
            items = line.split()
            index = items.index("fps,")
            info["fps"] = float(items[index - 1])
    return info

def get_data(path, size):
    width, height = size
    cmd = "ffmpeg -i {} -vf \"scale={}:{}:force_original_aspect_ratio=increase,crop={}:{}\" -sws_flags lanczos -f image2pipe -pix_fmt rgb24 -vcodec rawvideo -".format(path, width, height, width, height)
    data = subprocess.check_output(cmd, shell=True)
    video = np.fromstring(data, "uint8").reshape(-1, height, width, 3)
    video = np.transpose(video, (0, 2, 1, 3))
    return video
