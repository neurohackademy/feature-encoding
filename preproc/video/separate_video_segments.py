import glob
import shutil
from pathlib import Path

video1_segments = [
    [1, 480],
    [481, 6338],
    [6339, 6818],
    [6819, 12138],
    [12139, 12618],
    [12619, 17131],
    [17132, 17612],
    [17613, 19142],
    [19143, 19622],
    [19623, 21624],
    [21625, 22104],
]

video2_segments = [
    [1, 480],
    [481, 5922],
    [5923, 6403],
    [6404, 12609],
    [12610, 13089],
    [13090, 19071],
    [19072, 19550],
    [19551, 21552],
    [21553, 22032],
]
video3_segments = [
    [1, 480],
    [481, 4814],
    [4815, 5294],
    [5295, 9723],
    [9724, 10203],
    [10204, 15102],
    [15103, 15582],
    [15583, 19003],
    [19004, 19478],
    [19479, 21480],
    [21481, 21960],
]
video4_segments = [
    [1, 480],
    [481, 6056],
    [6057, 6536],
    [6537, 12053],
    [12054, 12533],
    [12534, 18658],
    [18659, 19142],
    [19143, 21144],
    [21145, 21624],
]

VID_DIR = "/home/ubuntu/hcp_data/jpg_256/"
video1 = "7T_MOVIE1_CC1_v2_256x256/"
video2 = "7T_MOVIE2_HO1_v2_256x256/"
video3 = "7T_MOVIE3_CC2_v2_256x256/"
video4 = "7T_MOVIE4_HO2_v2_256x256/"


videos = [video1, video2, video3, video4]
video_segments = [video1_segments, video2_segments, video3_segments, video4_segments]

split_videos_path = "/home/ubuntu/hcp_data/jpg_256/split_videos_256x256/"

csv_file = open(split_videos_path+"vid_list.csv", "w")


def get_frame_id(frame_name):
    frame_num = frame_name.split("frame")[-1].split(".jpg")[0]
    frame_num = int(frame_num)
    return frame_num


for idx in range(len(videos)):
    Path(split_videos_path + videos[idx]).mkdir(parents=True, exist_ok=True)
    frames = sorted(glob.glob(VID_DIR + videos[idx] + "*.jpg"), key=get_frame_id)
    for seg_num, segment in enumerate(video_segments[idx]):
        Path(split_videos_path + videos[idx] + videos[idx].replace("/","") + f"_seg_{seg_num}").mkdir(
            parents=True, exist_ok=True
        )
        csv_file.write(split_videos_path + videos[idx] + videos[idx].replace("/","") + f"_seg_{seg_num}"+"\n")
        for frame in frames[segment[0] - 1 : segment[1]]:
            shutil.copy(frame, split_videos_path + videos[idx] + videos[idx].replace("/","") + f"_seg_{seg_num}")
csv_file.close()