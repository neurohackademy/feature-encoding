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

def get_new_videoname(videoname):
	name, type = videoname.split(".")
	name = name + "_256x256"
	new_videoname = name + '.' + type
	return new_videoname

# %%
# main_dir = 'C:\\Users\\Spacetop\\Documents\\Moments_in_Time_Raw_v2\\Moments_in_Time_Raw'
main_dir = '/mnt/c/Users/Spacetop/Documents/Moments_in_Time_Raw_v2/Moments_in_Time_Raw'
in_dir = os.path.join(main_dir, 'training')
out_dir = os.path.join(main_dir,'resampled', 'training')

action_list = sorted(glob.glob(os.path.join(in_dir,'*'))) # asking, applauding

# %%
for action_dir in action_list:
    print("action dir: ", action_dir)
    action_name = os.path.basename(action_dir)
    video_list_tmp = glob.glob(os.path.join(action_dir,'*'))

    for index, old_videoname in enumerate(video_list_tmp):
        print("old_videoname: ", old_videoname)
        extension = os.path.basename(old_videoname).split('.')[-1]
        print("extension", extension)
        if extension == 'mp4':
            new_path = os.path.join(out_dir, action_name)
            Path(new_path).mkdir(parents = True, exist_ok = True)
            print("newpath: ", new_path)
            new_filename = os.path.join(new_path, f'video_{index:06d}.mp4')

            ## ffmpeg - crop and rescale
            stream = (ffmpeg.input(old_videoname).filter('scale', width = 365, height = 258).filter('crop', 256, 256).output(get_new_videoname(new_filename))).overwrite_output()
            ffmpeg.run(stream)

            # ffmpeg - split into frames
            frame_name = f'video_{index:06d}'
            frame_dir = os.path.join(new_path, f'video_{index:06d}')
            Path(frame_dir).mkdir(parents = True, exist_ok = True)
            frames = ffmpeg.input(os.path.join(new_path,f'video_{index:06d}_256x256.mp4')).filter('fps', fps='24', round='up').output(os.path.join(frame_dir,f'frames_%06d.jpg', ), start_number = 0 ).overwrite_output()
            ffmpeg.run(frames)
           
            # break
    # break
