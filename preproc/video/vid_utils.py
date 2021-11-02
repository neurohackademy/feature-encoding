import cv2
import numpy as np
from decord import VideoReader
from decord import cpu
# import torch

def describe_video(path):
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frameRate = int(cap.get(cv2.CAP_PROP_FPS))
    return frameRate, frameHeight, frameWidth, frameCount
    #print(f"Framerate: {frameRate}\nHeightxWidth: {frameHeight}x{frameWidth}\nN_frames: {frameCount}")
    
def read_mp4_video(file):
    images = list()
    vr = VideoReader(file, ctx=cpu(0))
    total_frames = len(vr)
    indices = np.linspace(0,total_frames-1,total_frames,dtype=np.int32)
    for seg_ind in indices:
        images.append(vr[seg_ind].asnumpy())
    # video = torch.Tensor(np.array(images))
                 #channels, frames, height, width
    # return video.permute(3,0,1,2)