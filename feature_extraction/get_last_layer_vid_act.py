# %%
import glob
# read in fmri data
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Read in video features

VID_DIR = "/home/ubuntu/hcp_data/jpg_256/split_videos_256x256/"
video1 = "7T_MOVIE1_CC1_v2_256x256"
video2 = "7T_MOVIE2_HO1_v2_256x256"
video3 = "7T_MOVIE3_CC2_v2_256x256"
video4 = "7T_MOVIE4_HO2_v2_256x256"

video_folders = [video4]

# vid1_feat = np.load(VID_DIR + video1, allow_pickle=True)
# vid2_feat = np.load(VID_DIR + video2, allow_pickle=True)
# vid3_feat = np.load(VID_DIR + video3, allow_pickle=True)
# vid4_feat = np.load(VID_DIR + video4, allow_pickle=True)

# %%

# Choose the `slowfast_r50` model
model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)
proj_layer = model.blocks[6].proj


# %%

df_labels = pd.read_csv("/home/ubuntu/feature-encoding/feature_extraction/kinetics_400_labels.csv")

labels = df_labels.name.values
print(labels.shape)


# %%

for video_folder in video_folders:
    video_segment_paths = glob.glob(VID_DIR + video_folder + "/*.npy")
    video_segment_paths = [path for path in video_segment_paths if "_act_ll.npy" not in path and "_classes.npy" not in path]
    
    video_raw_outputs = []
    video_classifications = []
    threshold = 0.80
    with torch.no_grad():
        for video_segment_path in sorted(video_segment_paths):
            vid_feat = np.load(video_segment_path, allow_pickle=True)
            print(vid_feat.shape)
            vid_feat_torch = torch.Tensor(vid_feat)
            vid_feat_last = proj_layer(vid_feat_torch)
            vid_classification = torch.sigmoid(vid_feat_last)
            vid_classification[vid_classification >= threshold] = 1
            vid_classification[vid_classification < threshold] = 0
            video_raw_outputs.append(vid_feat_last)
            video_classifications.append(vid_classification)
            np.save(
                (video_segment_path).replace(".npy", "_act_ll.npy"),
                vid_feat_last.detach().numpy(),
                allow_pickle=True,
            )
            np.save(
                (video_segment_path).replace(".npy", "_classes.npy"),
                vid_classification.detach().numpy(),
                allow_pickle=True,
            )

# %%
"""

frame = 1050
labels_for_frame = labels[video_classifications[0][frame] == 1]
path_frames = "/home/ubuntu/hcp_data/stimuli_jpg/7T_MOVIE1_CC1_v2_224x224"
path_frame = path_frames + "/frame" + str(frame + 1).zfill(4) + ".jpg"
img = plt.imread(path_frame)
plt.imshow(img)
plt.title(str(labels_for_frame))
plt.savefig("/home/ubuntu/tk_trial/viz_plot.jpg")
# %%
"""



# %%
