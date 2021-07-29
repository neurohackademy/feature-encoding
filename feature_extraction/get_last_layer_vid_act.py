# %%
import glob
# read in fmri data
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

# Read in video features
VID_DIR = "/home/ubuntu/hcp_data/stimuli_jpg/"
video1 = "7T_MOVIE1_CC1_v2_224x224_72.npy"
video2 = "7T_MOVIE2_HO1_v2_224x224_72.npy"
video3 = "7T_MOVIE3_CC2_v2_224x224_72.npy"
video4 = "7T_MOVIE4_HO2_v2_224x224_72.npy"

vid1_feat = np.load(VID_DIR + video1, allow_pickle=True)
vid2_feat = np.load(VID_DIR + video2, allow_pickle=True)
vid3_feat = np.load(VID_DIR + video3, allow_pickle=True)
vid4_feat = np.load(VID_DIR + video4, allow_pickle=True)

# %%

# Choose the `slowfast_r50` model
model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
proj_layer = model.blocks[6].proj


# %%

df_labels = pd.read_csv("kinetics_400_labels.csv")

labels = df_labels.name.values
print(labels.shape)


# %%

videos = [video1,video2,video3,video4]
video_raw_outputs = []
video_classifications = []
threshold = 0.80
with torch.no_grad():
    for video in videos:
        vid_feat = np.load(VID_DIR + video, allow_pickle = True)
        vid_feat_torch = torch.Tensor(vid_feat)
        vid_feat_last = proj_layer(vid_feat_torch)
        vid_classification = torch.sigmoid(vid_feat_last)
        vid_classification[vid_classification >= threshold] = 1
        vid_classification[vid_classification < threshold] = 0
        video_raw_outputs.append(vid_feat_last)
        video_classifications.append(vid_classification)
        np.save((VID_DIR + video).replace("_72","_72_last_layer"), vid_feat_last.detach().numpy(), allow_pickle=True)
        np.save((VID_DIR + video).replace("_72","_72_sigmoid"), vid_classification.detach().numpy(), allow_pickle=True)

# %%



frame = 1050
labels_for_frame = labels[video_classifications[0][frame]==1]
path_frames = "/home/ubuntu/hcp_data/stimuli_jpg/7T_MOVIE1_CC1_v2_224x224"
path_frame = path_frames+ "/frame"+ str(frame+1).zfill(4) + ".jpg"
img = plt.imread(path_frame)
plt.imshow(img)
plt.title(str(labels_for_frame))
# %%
