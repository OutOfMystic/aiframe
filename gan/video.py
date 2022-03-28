import os
import skvideo.io
import numpy as np
import cv2
import glob

frames = []
writer = skvideo.io.FFmpegWriter("outputvideo.mp4")
for filename in glob.glob('to_video/*.png'):
    frame = cv2.imread(filename)
    to_frames = [filename, frame]
    frames.append(to_frames)

sort_func = lambda elem: int(elem[0].split('.')[0].split('_')[-1])
frames.sort(key=sort_func)

for filename, frame in frames:
    writer.writeFrame(frame)
    print(filename)
writer.close()
