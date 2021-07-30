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
matplotlib.__version__
# import mayavi

# np.load('/home/ubuntu/hcp_data/jpg_256/split_videos_256x256/7T_MOVIE4_HO2_v2_256x256/7T_MOVIE4_HO2_v2_256x256_seg_0_72.npy').shape
# %%
def get_faces(faces, parc_idx):
    '''Returns a boolean array indicating if a faces from lies on the outer edge of the parcellation defined by the indices in parc_idx
    IN:
    faces      -   numpy ndarray of shape (n, 3), containing indices of the mesh faces
    parc_idx   -   indices of the vertices belonging to the region that is to be plotted
    '''
    faces_in_parc = np.array([np.isin(face, parc_idx) for face in faces])
    vertices_on_edge = np.intersect1d(np.unique(faces[faces_in_parc.sum(axis=1)==2]), parc_idx)
    faces_outside_edge = np.array([np.isin(face, vertices_on_edge) for face in faces]).sum(axis=1)
    faces_outside_edge = np.logical_and(faces_outside_edge > 0, faces_in_parc.sum(axis=1)<3)
    return faces_outside_edge

def modify_facecolors(new_color, faces_to_modify, axes):
    '''Modifies colors of mesh in axes by replacing all faces in faces_to_modify with new_color'''
    if isinstance(new_color, str):
        new_color = np.array(clr.to_rgb(color)+(1.,))
    poly = axes.collections[0]
    # fcolors = poly._facecolor
    fcolors = poly._facecolors3d
    # _facecolor
    fcolors[faces_outside] = np.array(new_color)
    poly._facecolors3d = fcolors
    return axes

    
# %%
# path parameters and fsaverage
main_dir   = '/home/ubuntu'
result_dir = os.path.join(main_dir, 'results', 'cross_subject')
fsaverage = datasets.fetch_surf_fsaverage(mesh='fsaverage5')
# /home/ubuntu/tk_trial/LSTM/results/cross_subject/groupbeta_0.nii.gz

# group_fname = os.path.join(result_dir, "groupbeta_0.nii.gz")
group_fname = '/home/ubuntu/tk_trial/LSTM/results/cross_subject/groupbeta_0.nii.gz'
group = image.load_img(group_fname) 
group0_surf_R = surface.vol_to_surf(group, fsaverage.pial_right)
group0_surf_L = surface.vol_to_surf(group, fsaverage.pial_left)

group_fname = '/home/ubuntu/tk_trial/LSTM/results/cross_subject/groupbeta_1.nii.gz'
group = image.load_img(group_fname) 
group1_surf_R = surface.vol_to_surf(group, fsaverage.pial_right)
group1_surf_L = surface.vol_to_surf(group, fsaverage.pial_left)

group_fname = '/home/ubuntu/tk_trial/LSTM/results/cross_subject/groupbeta_2.nii.gz'
group = image.load_img(group_fname) 
group2_surf_R = surface.vol_to_surf(group, fsaverage.pial_right)
group2_surf_L = surface.vol_to_surf(group, fsaverage.pial_left)

group_fname = '/home/ubuntu/tk_trial/LSTM/results/cross_subject/groupbeta_3.nii.gz'
group = image.load_img(group_fname) 
group3_surf_R = surface.vol_to_surf(group, fsaverage.pial_right)
group3_surf_L = surface.vol_to_surf(group, fsaverage.pial_left)




# %%
# outline
# TPJ outline _____________________________________________________________________________________
action_uniformity_fname = '/home/ubuntu/feature-encoding/viz/brain_plot/action_uniformity-test_z_FDR_0.01.nii.gz'
neurosynth_thres = 0.
thres = 1.

fig, axes = plt.subplots(1,1,subplot_kw={'projection':'3d'}, figsize=(9, 6))
plotting.plot_surf_stat_map(fsaverage.infl_right, group0_surf_R, hemi='right',
                            title='Surface right hemisphere', colorbar=True,
                            threshold=1., bg_map=fsaverage.sulc_right, 
                             figure = fig, axes=axes)
plotting.show()

# c = plotting.plot_surf(fsaverage.infl_right,empathy_rev_surf_R, hemi='right',
#                             title='Surface right hemisphere', 
#                             bg_map=fsaverage.sulc_right, alpha = alpha_thres, threshold = thres,
#                            cmap = 'BuPu_r', figure = fig, axes=axes, avg_method = 'median')
action_ns = image.threshold_img(action_uniformity_fname, 
    threshold=neurosynth_thres, 
    copy=False)

texture = surface.vol_to_surf(action_ns, fsaverage.pial_right)

# plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
#                             title='Surface right hemisphere', colorbar=False,
#                             bg_map=fsaverage.sulc_right, alpha = 0., threshold=thres, 
#                             figure = fig, axes=axes)
# https://mjboos.github.io/Nilearn-surface-contours/

# coords, faces = surface.load_surf_mesh(fsaverage.infl_right)

# load vertex coordinates and face indices that specify the surface mesh
coords, faces = load_surf_mesh(fsaverage.infl_right)
destrieux_atlas = datasets.fetch_atlas_surf_destrieux()
parcellation = destrieux_atlas['map_right']

# these are the regions we want to outline
regions = [b'G_pariet_inf-Angular',
 b'G_precentral',  b'G_postcentral']
regions_idx = [np.where(np.array(destrieux_atlas['labels'])==region)[0]
               for region in regions]
colors = ['g', 'magenta', 'cyan']

patch_list = []
for reg_name, reg_i, color in zip(regions, regions_idx, colors):
    parc_idx = np.where(parcellation==reg_i)[0]
    faces_outside = get_faces(faces, parc_idx)
    modify_facecolors(color, faces_outside, axes)
    patch_list.append(mpatches.Patch(color=color, label=reg_name))
fig.legend(handles=patch_list)


# reg_name = "action"
# patch_list = []
# parc_idx = np.where(texture>neurosynth_thres)[0]
# faces_outside = get_faces(faces, parc_idx)
# # color = 'cyan'
# # modify_facecolors(color, faces_outside, axes)

# poly = axes.collections[0]
# fcolors = poly._facecolors3d
# new_color = np.array(clr.to_rgb('k')+(1.,))
# fcolors[faces_outside] = np.array(new_color)
# poly._facecolors3d = fcolors


# regions = [b'G_pariet_inf-Angular',
#  b'G_precentral',  b'G_postcentral']
# regions_idx = [np.where(np.array(destrieux_atlas['labels'])==region)[0]
#                for region in regions]
# colors = ['g', 'magenta', 'cyan']

# patch_list = []
# for reg_name, reg_i, color in zip(regions, regions_idx, colors):
#     parc_idx = np.where(parcellation==reg_i)[0]
#     faces_outside = get_faces(faces, parc_idx)
#     modify_facecolors(color, faces_outside, axes)
#     patch_list.append(mpatches.Patch(color=color, label=reg_name))
# fig.legend(handles=patch_list)



plt.show()

# %%
#build single surface with all maps ______________________________________
maps = [  group0_surf_R, group1_surf_R, group2_surf_R, group3_surf_R]#, TPJsurf_R]
cmaps = [ "red", "#0000FF", "lime", "yellow"]#, "black"]
bg_map = fsaverage.sulc_right
#derived from plot_surf in nilearn
#specify mesh base ______________________________________
mesh = surface.load_surf_mesh(fsaverage['infl_right'])
coords, faces = mesh[0], mesh[1] #coords are 3-d coords of vertices, faces are 3 indices into the coords defining a triangle in 3-D space
limits = [coords.min(), coords.max()]
elev = 0.
azim = -15. #195.  #-15.
alpha = 0.1 # 1.0
darkness = 0.2
face_colors = np.ones((faces.shape[0], 4))
threshold = 0.0
#specify figure ______________________________________
plt.rcParams["figure.figsize"] = [16,10]
fig, axes = plt.subplots(1,1, subplot_kw={'projection': '3d'})
axes.view_init(elev=elev, azim=azim)
axes.set_axis_off()

#plot uncolored surface ______________________________________
p3dcollec = axes.plot_trisurf(coords[:, 0], coords[:, 1], coords[:, 2], triangles=faces, linewidth=0., 
                              antialiased=False, color='white')
#load bg map 

bg_data = surface.load_surf_data(bg_map)
if bg_data.shape[0] != coords.shape[0]:
    raise ValueError('The bg_map does not have the same number '
                             'of vertices as the mesh.')
# bg_data = surface.load_surf_data(fsaverage['sulc_right'])
bg_faces = np.mean(bg_data[faces], axis=1)
bg_faces = bg_faces - bg_faces.min()
bg_faces = bg_faces / bg_faces.max()
bg_faces *= darkness #compress range to limit darkness
face_colors = plt.cm.gray_r(bg_faces)


surf_face_colors = np.zeros_like(face_colors)
for i in range(len(maps)):
    surf_map_data = surface.load_surf_data(maps[i]) #data for each vertex (3-D locations in coords, len=coords)
    surf_map_faces = np.mean(surf_map_data[faces], axis=1)  #average of data values from the 3 defining vertices (len=faces)
    vmin = 0#np.nanmin(surf_map_faces) # 0
    vmax = 1.96#np.nanmax(surf_map_faces) # 1.96
    surf_map_faces = surf_map_faces - vmin
    #surf_map_faces = surf_map_faces / (vmax - vmin)
    floor_inds = np.where(np.abs(surf_map_faces) < 0.2)
    surf_map_faces[floor_inds] = 0.0
    #cap_inds = np.where(np.abs(surf_map_faces) >= 1.96)
    cap_inds = np.where(np.abs(surf_map_faces) >= 200)
    surf_map_faces[cap_inds] = 1.0
    kept_indices = np.where(np.abs(surf_map_faces) >= threshold)[0]
    #cmap = plt.cm.get_cmap(cmaps[i]) #get cmap for this neuromap by cmaps string
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white", cmaps[i]], N=300, gamma = 10)
    try: 
        surf_face_colors[kept_indices] +=  cmap(surf_map_faces[kept_indices])
    except: 
        surf_face_colors[kept_indices] = cmap(surf_map_faces[kept_indices])
surf_face_colors -= surf_face_colors.min()
surf_face_colors /= surf_face_colors.max()
#norm = colors.Normalize(vmin=vmin, vmax=vmax)

#add background
face_colors *= surf_face_colors 

#range correct
#face_colors -= face_colors.min()
#face_colors /= face_colors.max()
#cap at 1 
#cap_inds = np.where(np.abs(face_colors) >= 1.0)
#face_colors[cap_inds] = 1.0 

#set face colors in figure
p3dcollec.set_facecolors(face_colors)

# TPJ outline _____________________________________________________________________________________

action_uniformity_fname = '/home/ubuntu/feature-encoding/viz/brain_plot/action_uniformity-test_z_FDR_0.01.nii.gz'
neurosynth_thres = 0.
thres = 1.
action_ns = image.threshold_img(action_uniformity_fname, 
    threshold=neurosynth_thres, 
    copy=False)

texture = surface.vol_to_surf(action_ns, fsaverage.pial_right)


plotting.plot_surf_stat_map(fsaverage.infl_right, texture, hemi='right',
                            title='Surface right hemisphere', colorbar=False,
                            bg_map=fsaverage.sulc_right, alpha = 1., threshold=thres, 
                            figure = fig, axes=axes, bg_on_data = True)
# https://mjboos.github.io/Nilearn-surface-contours/
coords, faces = surface.load_surf_mesh(fsaverage.infl_right)



# 

# c = plotting.plot_surf(fsaverage.infl_right,empathy_rev_surf_R, hemi='right',
#                             title='Surface right hemisphere', 
#                             bg_map=fsaverage.sulc_right, alpha = alpha_thres, threshold = thres,
#                            cmap = 'BuPu_r', figure = fig, axes=axes, avg_method = 'median')


reg_name = "TPJ"
patch_list = []
parc_idx = np.where(texture>neurosynth_thres)[0]
faces_outside = get_faces(faces, parc_idx)
#modify_facecolors(color, faces_outside, axes)

poly = axes.collections[0]
fcolors = poly._facecolors
new_color = np.array(clr.to_rgb('k')+(1.,))
fcolors[faces_outside] = np.array(new_color)
poly._facecolors = fcolors
# TPJ outline _____________________________________________________________________________________


# plt.savefig(os.path.join(fig_dir, "overlay"+str(i)+".png"), dpi=500)
# legend _____________________________________________________________________________________
# l_cmaps = [ "red", "blue", "lime", "yellow"]
# l_keyword = ["attention","memory","objects","language"]
# legend_elements = [ 

#                   Line2D([0], [0], marker='o', color='w', label=l_keyword[0],
#                           markerfacecolor=l_cmaps[0], markersize=15),
#                   Line2D([0], [0], marker='o', color='w', label=l_keyword[1],
#                           markerfacecolor=l_cmaps[1], markersize=15),
#                   Line2D([0], [0], marker='o', color='w', label=l_keyword[2],
#                           markerfacecolor=l_cmaps[2], markersize=15),
#                   Line2D([0], [0], marker='o', color='w', label=l_keyword[3],
#                           markerfacecolor=l_cmaps[3], markersize=15),
#                   Line2D([0], [0], color='k', lw=4, label='TPJ')]

# fig.legend(handles=legend_elements)
plt.show()