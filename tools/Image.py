import subprocess
import natsort
import os

MAGICK = "magick"
if os.name == "posix":
    MAGICK = "convert"

def images2gif(seq, target, fps=5):
    path, name = os.path.split(seq)
    image_files = [os.path.join(path, file) for file in next(os.walk(path))[2] if file.startswith(name) if os.path.splitext(file)[1] != ".gif"]
    image_files = natsort.natsorted(image_files)
    subprocess.call(MAGICK + " {} -loop 0 -delay {} {}".format(
        " ".join(image_files),
        int(100 / fps),
        target,
    ), shell=True)
