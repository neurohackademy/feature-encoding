# old version > output 224 x 224
INDIR=/home/ubuntu/data/raw
OUTDIR=/home/ubuntu/data/resampled
FILENAME="7T_MOVIE4_HO2_v2"
ffmpeg -i ${INDIR}/${FILENAME}.mp4 -filter:v "scale=320:225, crop=224:224" -c:a copy ${OUTDIR}/${FILENAME}_224x224.mp4

# new version > output 256 x 256
INDIR=/home/ubuntu/data/raw
OUTDIR=/home/ubuntu/data/resampled
FILENAME="7T_MOVIE4_HO2_v2"
ffmpeg -i ${INDIR}/${FILENAME}.mp4 -filter:v "scale=365:258, crop=256:256" -c:a copy ${OUTDIR}/${FILENAME}_256x256.mp4

# converting video to jpeg
INDIR=/home/ubuntu/data/raw
OUTDIR=/home/ubuntu/data/resampled
FILENAME="7T_MOVIE1_CC1_v2_256x256"
mkdir -p ${OUTDIR}/${FILENAME}
ffmpeg -i ${INDIR}/${FILENAME}.mp4 -r 24 ${OUTDIR}/${FILENAME}/frame%06d.jpg -hide_banner
