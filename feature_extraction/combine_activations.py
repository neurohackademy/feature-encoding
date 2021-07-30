# %%
import glob
# read in fmri data
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

def sort_split_vidname(name):
    num = name.split("seg_")[-1].split("_")[0]
    return int(num)

video1 = "7T_MOVIE1_CC1_v2_256x256"
video2 = "7T_MOVIE2_HO1_v2_256x256"
video3 = "7T_MOVIE3_CC2_v2_256x256"
video4 = "7T_MOVIE4_HO2_v2_256x256"

video_folders = [video1, video2, video3, video4]


for video in video_folders:
    path_activations = f"/home/ubuntu/hcp_data/jpg_256/split_videos_256x256/{video}"

    for kind in ["classes","act_ll"]:
        activations_filepaths = sorted(glob.glob(path_activations + f"/*_{kind}.npy"), key = sort_split_vidname)

        activations = [np.load(path, allow_pickle=True) for path in activations_filepaths]

        dims = [activation.shape[0] for activation in activations]

        concated_activations = np.concatenate(activations, axis = 0)
        print(concated_activations.shape)

        np.save(path_activations+f"_all_{kind}.npy", concated_activations, allow_pickle= True)