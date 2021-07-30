# Feature Encoding (Neurohackademy Project 2021)

**Project Goals**: (1) Use Facebook's [SlowFast](https://github.com/facebookresearch/SlowFast) video classification model to extract features (e.g., kinetics) from video stimuli; (2) Build a feature encoding model using LASSO regression to examine which brain regions encode information about the observed features

**Project Contributors** (alphabetical): [Alberto Mario Ceballos Arroyo](https://github.com/alceballosa), [Tomas D'Amelio](https://github.com/tomdamelio), [Heejung Jung](https://github.com/jungheejung), [Adriana Mendez Leal](https://github.com/asmendezleal), [Shawn Rhoads](https://github.com/shawnrhoads), [Shelby Wilcox](https://github.com/shelbywilcox), [Tiankang Xie](https://github.com/TiankangXie)

<hr>

## Getting started

### 1. Clone this repository 

**Using Git**</br>
In your terminal, use command: `git clone https://github.com/neurohackademy/feature-encoding`

Once the download completes, change your directory to `feature-encoding` (for example, using the command `cd feature-encoding`). Explore the directory structure and see how it is organized at [the end of this document](#file-structure).

**Using the file explorer / finder**</br>
Alternatively, you can download this [.zip file](https://github.com/neurohackademy/feature-encoding/archive/master.zip) and unzip it in a directory somewhere on your computer. Then, open "Anaconda Prompt" (Windows) or Terminal (MacOS) and change your directory. For example: 
- MacOS: cd ~/Desktop/feature-encoding-master
- Windows: cd "C:\Users\USERNAME\Desktop\feature-encoding-master"

### 2. Install Anaconda

You can follow [these instructions](https://docs.anaconda.com/anaconda/install/) to install Anaconda for your operating system.

### 3. Create a new Anaconda environment

Create the slowfast environment using this command: `conda env create -f conda_env_slowfast.yml`.

<hr>

## Preprocessing data

We have two main types of data: (1) the video data (`*.mp4` files) and (2) neuroimaging (fMRI) data (`*.nii.gz` files). The neuroimaging data we use are from the Human Connectome Project, in which participants watch short clips of videos. We will need to run these data through a few steps. <i>Note: these steps assume that all neuroimaging data have been preprocessed using standardized pipelines (i.e., slice timing correction, realignment, normalization to standard template, smoothing, and denoising)</i>.

### 1. Preprocessing steps for video data

To run the video stimuli through SlowFast, we need to resize our data to be 256 pixels x 256 pixels and then split up the entire video into frames (e.g., 24 frames/second x 921 seconds = 22104 frames). Use the `preproc/video/preproc_vid.sh` shell script to accomplish this.

In addition, because our stimuli consist of blocks of video clips, we also separate frames by video block. Use the `preproc/video/separate_video_segments.py` Python script to accomplish this.

### 2. Preprocessing steps for fMRI data

The neuroimaging data are 4-dimensional: 113 x 136 x 113 voxels (1.6mm<sup>3</sup>) collected over intervals of 1 second. However, we are not interested in some of the spatial regions (e.g., white matter, cerebrospinal fluid). Therefore, we mask each 4D image to only include gray matter using the `preproc/neuro/masker.py` Python script (if you are able to use parallelization, `preproc/neuro/masker_parallel.py` also gets the job done much quicker).

To futher reduce the spatial dimensionality, we also downsample our voxel size from 1.6<sup>3</sup> to 3.0<sup>3</sup> (61 x 74 x 61 voxels) using the `preproc/neuro/resampledat.py` Python script.

<hr>

## Running Facebook's SlowFast algorithm

### 1. Adjusting SlowFast parameter settings

If you would like to configue the model parameters, please edit them in this file: `feature_extraction/slowfast/config/SLOWFAST_8x8_R50.yaml`

To run all the video stimuli through SlowFast, navigate to the `feature_extraction/slowfast/` directory and run this command: `python run_net.py --cfg ./configs/SLOWFAST_8x8_R50.yaml`

### 2. Running the outputs through the final layer

To extract classes at each frame of the video, we must pass the outputs from SlowFast through the final layer. We can match these classes to their [labels](https://github.com/neurohackademy/feature-encoding/blob/main/feature_extraction/kinetics_400_labels.csv) by first running the `feature_extraction/combine_activations.py` Python script and then running the `feature_extraction/get_last_layer_vid_act.py` Python script.

<hr>

## Building an encoding model

### 1. Dimension reduction + LASSO

We will use a least absolute shrinkage and selection operator (LASSO) encoding model to examine which brain regions encode information about the observed features. 

To reduce the number of classes (400 total) from SlowFast, we ran a principle component analysis to derive the component loadings that explained 95% of the variance. Then we use these components to predict BOLD activity during the video watching. Use the `models/linear/linear_model.py` Python script to accomplish this.

### 2. Group-level inference

For group-level inference, we test which voxel-wise beta weights were different from zero on average across participants using a two-sided t-test. Use the `models/linear/stat_t_test.py` Python script to accomplish this.

<hr>

## Plotting results

### 1. Visualizing the class data

Word clouds are one method to visualize the class loadings for each of the principle components. To view the world cloud plots, use the Python script: `viz/pc_loadings/step01_mergecsv.py`

### 2. Visualizing the neural data

We plot the thresholded neuroimaging results from the different principle components along with the features that clustered within each component. We suply two Python scripts for accomplishing this: `viz/brain_plot/code/plot_surface.py` and `viz/brain_plot/code/plot_surface_tmap.py`.

<hr>

## Bonus: Building a Long Short-Term Memory Encoding Model

Python scripts: `models/lstm/LSTMNet.py`, `models/lstm/LSTM_loader.py`, `models/lstm/LSTM_main.py`

<hr>

## File structure
```
.
├── LICENSE
├── README.md
├── RESOURCES.md
├── conda_env_slowfast.yml
├── data
│   └── video
│       ├── frames
│       │   └── 7T_MOVIE1_CC1_v2
│       │       └── frame000012.jpg
│       ├── raw
│       │   └── 7T_MOVIE1_CC1_v2.mp4
│       └── resampled
│           └── 7T_MOVIE1_CC1_v2_224x224.mp4
├── feature_extraction
│   ├── combine_activations.py
│   ├── get_last_layer_vid_act.py
│   ├── kinetics_400_labels.csv
│   └── slowfast
│       ├── config
│       │   ├── SLOWFAST_8x8_R50.yaml
│       │   └── SLOWFAST_8x8_R50_244.yaml
│       └── enviro_setup.md
├── models
│   ├── linear
│   │   ├── README.md
│   │   ├── linear_model.py
│   │   ├── plot_brain.py
│   │   └── stat_t_test.py
│   └── lstm
│       ├── LSTMNet.py
│       ├── LSTM_loader.py
│       ├── LSTM_main.py
│       └── README.md
├── preproc
│   ├── neuro
│   │   ├── masker.py
│   │   ├── masker_parallel.py
│   │   └── resample_voxel.py
│   └── video
│       ├── preproc_vid.sh
│       └── separate_video_segments.py
└── viz
    ├── brain_plot
    │   ├── code
    │   │   ├── plot_surface.py
    │   │   └── plot_surface_tmap.py
    │   ├── derivatives
    │   │   ├── surface-groupbeta_pc-*.jpg
    │   └── input
    │       └── action_uniformity-test_z_FDR_0.01.nii.gz
    └── pc_loadings
        ├── kinetics_400_labels.csv
        ├── kinetics_pc.csv
        ├── step01_mergecsv.py
        └── twovid_pca.npy
```
