# RESAMPLED: confirm which frames were processed
# TRAINING: delete if they have been (.\Moments_in_Time_Raw\training)
# TODO: create resample - training key list. 
# one thing that we're not for sure: 
# - RESAMPLED video and TRAINING video order. 
# - glob does not sort. 
# - concerned that the video index in reample doesn't match the training videos

# %% 
"""
convert videos to frames
also rename videos from original algonaut folders to video_000001.mp4
"""
# TODO: 
# [x] 1. grab list of folders from INDIR, 
# [x] 2. make new dir in OUTDIR resampled
# [x] 3. change folder name to indices instead of original gibberish folder name. 

# %% libraries
import os, glob, re, shutil
from pathlib import Path
import ffmpeg
import tarfile
from tqdm import tqdm
import pandas as pd

def get_new_videoname(videoname):
	name, type = videoname.split(".")
	name = name + "_256x256"
	new_videoname = name + '.' + type
	return new_videoname

# %%
main_dir = '/mnt/c/Users/Spacetop/Documents/Moments_in_Time_Raw_v2/Moments_in_Time_Raw'
in_dir = os.path.join(main_dir, 'training')
out_dir = os.path.join(main_dir,'resampled', 'training')

action_list = glob.glob(os.path.join(in_dir,'*')) # asking, applauding
dict1={}
feat_list=[]
# video_key = pd.DataFrame(columns = ['algonauts_fname', 'video_index'])
for action_dir in tqdm(action_list):
    # print("action dir: ", action_dir)
    action_name = os.path.basename(action_dir) # adult+female+singing
    video_list_tmp = glob.glob(os.path.join(action_dir,'*')) # adult+female+singing/peeks-www_k_to_keek_0iwteab_20.mp4

    for index, old_videoname in enumerate(video_list_tmp):
        # print("old_videoname: ", old_videoname)
        extension = os.path.basename(old_videoname).split('.')[-1]
        # print("extension", extension)
        if extension == 'mp4':
            new_path = os.path.join(out_dir, action_name)
            frame_dir = os.path.join(new_path, f'video_{index:06d}')
            if os.path.exists(frame_dir) and len(os.listdir(frame_dir)) > 5:
                dict1 = dict([('algonauts_fname',old_videoname),
                    ('video_index',f'video_{index:06d}')])
                feat_list.append(dict1)
                            # video_key.append(( f'video_{index:06d}', old_videoname ))
                # compress video
                with tarfile.open(old_videoname, "w:gz") as tar:
                    tar.add(action_dir, arcname=os.path.basename(action_dir))
            else:
                continue
feat_df=pd.DataFrame(feat_list)
feat_df.to_csv('/mnt/c/Users/Spacetop/Documents/heejung/feature-encoding/preproc/video/video_key.csv')


           
            # break
    # break


# %%
