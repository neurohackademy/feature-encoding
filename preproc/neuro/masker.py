import os, nibabel as nib, nilearn
from glob import glob
from nilearn.input_data import NiftiMasker

data = glob('/home/ubuntu/hcp_data/ten_subjects/*/MNINonLinear/Results/tfMRI_MOVIE*_7T_*/tfMRI_MOVIE*_7T_*_hp2000_clean.nii.gz')

masker = NiftiMasker(mask_img='/home/ubuntu/tk_trial/hcp_MNImask_1.6mm.nii.gz')
#masker = NiftiMasker(mask_img='/home/ubuntu/tk_trial/hcp_MotionMask_1.6mm.nii.gz')

mask = nib.load('/home/ubuntu/tk_trial/hcp_MNImask_1.6mm.nii.gz')

for subj in data:

    file_dir = os.path.dirname(subj)
    filename = os.path.basename(subj)
    outname = os.path.join(file_dir, filename[:-7]+'_masked'+'.nii.gz')

    if not os.path.isfile(outname):
        print('running- ', outname)

        subj_data = nib.load(subj)
        z = masker.fit_transform(subj_data)

        subj_data_masked = nilearn.masking.unmask(z, mask)
        nib.save(subj_data_masked, outname)

    else:
        print('skipping- ', outname)
        continue

# subj_arr_masked = subj_data_masked.get_fdata()

# plt.imshow(subj_arr_masked[:,:,60,100])
# plt.show()
 
