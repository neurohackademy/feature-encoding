# %%
# use alberto code: models\models_alberto\models.py
# grab one video path and see if it works
import models as algo_models

# wrapped_model = algo_models.select_model(model_name)()
# activations = algo_models.get_activations(wrapped_model, video_path, True)
# for activation in activations.keys():
#     print(activation, activations[activation].shape)

# %% 
# vid
# 
# eo_path = 'C:\Users\Spacetop\Documents\Moments_in_Time_Raw_v2\Moments_in_Time_Raw\resampled\training\applauding'
video_path = '/mnt/c/Users/Spacetop/Documents/Moments_in_Time_Raw_v2/Moments_in_Time_Raw/resampled/training/ascending'

model_name = 'slowfast'
# TODO: need to train model on actual algonauts videos
# currently, we're only grabbing pre-trained weights in this code.
wrapped_model = algo_models.select_model(model_name)()
activations = algo_models.get_activations(wrapped_model, video_path, True)
print(activations)
for activation in activations.keys():
    print(activation, activations[activation].shape)