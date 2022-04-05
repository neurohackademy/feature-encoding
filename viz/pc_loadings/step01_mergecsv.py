# library
# %%
import numpy as np
import pandas as pd
import os
# from PIL import Image
# from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# %%
git_dir = '/home/ubuntu/feature-encoding'
fname =  os.path.join(git_dir, 'viz', 'pc_loadings', 'twovid_pca.npy')
pca = pd.DataFrame(np.load(fname).T)
print(pca.shape)
# rename columns
num_cols = len(list(pca))
rng = range(0, num_cols)
new_cols = ['pc_' + str(i).zfill(2) for i in rng] 
pca.columns = new_cols[:num_cols]
pca.head()

# %%
# load kinteics labels
labels = os.path.join(git_dir, 'viz', 'pc_loadings', 'kinetics_400_labels.csv')
label_df = pd.read_csv(labels)

print("shape: ", label_df.shape)
label_df.head()


# %%

new_df = label_df.merge(pca, left_index = True, right_index = True)
new_df.to_csv(os.path.join(git_dir, 'viz', 'pc_loadings','kinetics_pc.csv'))
new_df.head()
new_df.shape
