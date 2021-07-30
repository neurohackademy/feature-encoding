# %%
from nilearn import surface
from nilearn import datasets
from nilearn import plotting
from nilearn import image
from nilearn import regions
from nilearn.surface import load_surf_mesh
import os
import pylab as plt
import numpy as np
from matplotlib import colors as clr
import matplotlib.patches as mpatches
import matplotlib
import glob
import re
matplotlib.__version__

# %%
# path parameters and fsaverage
main_dir   = '/home/ubuntu/'
result_dir = os.path.join(main_dir, 'tk_trial', 'LSTM','results', 'cross_subject')
save_dir = os.path.join(main_dir, 'feature-encoding', 'viz', 'brain_plot', 'derivatives')
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
# /home/ubuntu/tk_trial/LSTM/results/cross_subject/groupbeta_0.nii.gz
# group_fname = os.path.join(result_dir, "groupbeta_0.nii.gz")
group_fnames = glob.glob(os.path.join(result_dir, 'groupbeta_*.nii.gz'))
# group_fnames
#%%
sorted(group_fnames)


# %%
for fname in sorted(group_fnames):
    group = image.load_img(fname) 
    pc_num = re.findall(r'\d+', os.path.basename(fname))
    group_surf_R = surface.vol_to_surf(group, fsaverage.pial_right)
    group_surf_L = surface.vol_to_surf(group, fsaverage.pial_left)

    fig, axes = plt.subplots(1,1,subplot_kw={'projection':'3d'}, figsize=(9, 6))
    group_surf = plotting.plot_surf_stat_map(fsaverage.infl_right, group_surf_R, hemi='right',
                                title='PC no.' + str(pc_num[0]).zfill(2), colorbar=True,
                                threshold=1., bg_map=fsaverage.sulc_right, 
                                figure = fig, axes=axes)
    #

    # save file
    pc_num = re.findall(r'\d+', os.path.basename(fname))
    plt.savefig(os.path.join(save_dir, 'surface-groupbeta_pc-' +str(pc_num[0]).zfill(2) + '.jpg'))
    plotting.show()

