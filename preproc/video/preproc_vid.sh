#!/bin/bash
# # old version > output 224 x 224
# INDIR=/home/ubuntu/data/raw
# OUTDIR=/mnt/c/Users/Spacetop/Documents/Moments_in_Time_Raw_v2/Moments_in_Time_Raw/resampled
# FILENAME="7T_MOVIE4_HO2_v2"
# ffmpeg -i ${INDIR}/${FILENAME}.mp4 -filter:v "scale=320:225, crop=224:224" -c:a copy ${OUTDIR}/${FILENAME}_224x224.mp4
# new version > output 256 x 256
# _______________________
INDIR="/mnt/c/Users/Spacetop/Documents/Moments_in_Time_Raw_v2/Moments_in_Time_Raw/training/applauding"
OUTDIR="/mnt/c/Users/Spacetop/Documents/Moments_in_Time_Raw_v2/Moments_in_Time_Raw/resampled/training/applauding"

for FULLFILE in $(ls ${INDIR})
do
mkdir -p ${OUTDIR}/${FILENAME}
FILENAME="${FULLFILE%.*}"
ffmpeg -i ${INDIR}/${FILENAME}.mp4 -filter:v "scale=365:258, crop=256:256" -c:a copy ${OUTDIR}/${FILENAME}/${FILENAME}_256x256.mp4
done
# _______________________

# /mnt/c/Users/Spacetop/Documents/Moments_in_Time_Raw_v2/Moments_in_Time_Raw/resampled/training/applauding/D9draVjaGQI_5/D9draVjaGQI_5_256x256
# converting video to jpeg
# INDIR=/mnt/c/Users/Spacetop/Documents/Moments_in_Time_Raw_v2/Moments_in_Time_Raw/training/applauding
# OUTDIR=/mnt/c/Users/Spacetop/Documents/Moments_in_Time_Raw_v2/Moments_in_Time_Raw/resampled/training/applauding

# for FULLFILE in $(ls ${OUTDIR})
# do
# # ..\Moments_in_Time_Raw\resampled\training\applauding\D9draVjaGQI_5\frames
# FILENAME="${FULLFILE%.*}"
# mkdir -p ${OUTDIR}/${FILENAME}/frames
# ffmpeg -i ${OUTDIR}/${FILENAME}/${FILENAME}_256x256.mp4 -r 24 ${OUTDIR}/${FILENAME}/frames/frame%06d.jpg -hide_banner
# done
