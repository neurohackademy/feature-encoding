import torch
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
)
import torchvision
import vid_utils
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)

from tqdm import tqdm
import os
import numpy as np 
import gc

class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors. 
    """
    def __init__(self,max_frames,stride_fast,slowfast_alpha):
        super().__init__()
        self.max_frames = max_frames
        self.stride_fast = stride_fast
        self.slowfast_alpha = slowfast_alpha
        
    def forward(self, frames: torch.Tensor):
        frames = frames[:,0:self.max_frames,:,:]
        fast_pathway = torch.index_select(
            frames,
            1,
            torch.linspace(
                0, frames.shape[1] - 1, frames.shape[1] // self.stride_fast
            ).long(),
        )
        # Perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(
            fast_pathway,
            1,
            torch.linspace(
                0, fast_pathway.shape[1] - 1, fast_pathway.shape[1] // self.slowfast_alpha
            ).long(),
        )
        frame_list = [slow_pathway, fast_pathway]
        return frame_list

class SlowfastWrapper:
    def __init__(self):
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        self.model = self.model.cuda().eval()
        self.layers = [ 
                #["slow_pathway_1", 2, self.model.blocks[1].multipathway_blocks[0].res_blocks[-1]],
                ["fast_pathway_1", 2, self.model.blocks[1].multipathway_blocks[1].res_blocks[-1]],
                #["slow_pathway_2", 2, self.model.blocks[2].multipathway_blocks[0].res_blocks[-1]],
                ["fast_pathway_2", 2, self.model.blocks[2].multipathway_blocks[1].res_blocks[-1]],
                ["slow_pathway_3", 2, self.model.blocks[3].multipathway_blocks[0].res_blocks[-1]],
                ["fast_pathway_3", 2, self.model.blocks[3].multipathway_blocks[1].res_blocks[-1]],
                ["slow_pathway_4", 2, self.model.blocks[4].multipathway_blocks[0].res_blocks[-1]],
                ["fast_pathway_4", 2, self.model.blocks[4].multipathway_blocks[1].res_blocks[-1]],
                ["block_5", -1, self.model.blocks[5]],
                ["block_6_proj", -1, self.model.blocks[6].proj]
        ]
        self.layer_names = [layer[0] for layer in self.layers]
        self.batch_model = False
        
        
    def preprocess_algonauts_video(self, video_path):
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256
        max_frames = 64
        frames_per_second = 30
        stride_slow = 8
        slowfast_alpha = 4
        stride_fast = stride_slow//slowfast_alpha
        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(max_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size),
                    PackPathway(max_frames, stride_fast, slowfast_alpha)
                ]
            ),
        )
        start_sec = 0
        video = {'video': vid_utils.read_mp4_video(video_path)}
        video_data = transform(video)
        inputs = video_data["video"]
        inputs = [i.cuda()[None, ...] for i in inputs]
        return inputs
    
    def get_activations(self, video_path, flatten = True):
        activations_dir = {}
        def get_activation(name):
            def hook(model, input, output):
                activations_dir[name] = output.detach().cpu().numpy()
            return hook
        # Set up hooks for getting the activations
        handles = []
        for layer in self.layers:
            handle = layer[-1].register_forward_hook(get_activation(layer[0]))
            handles.append(handle)
        inputs = self.preprocess_algonauts_video(video_path)
        _ = self.model(inputs)

        for layer in self.layers:
            if type(layer[1]) == list or layer[1] > 0:
                activations_dir[layer[0]] =activations_dir[layer[0]].mean(axis=layer[1])
            if flatten:
                activations_dir[layer[0]]=activations_dir[layer[0]].flatten()
        for handle in handles:
            handle.remove()
        return activations_dir

class ResNet50Wrapper():
    def __init__(self):
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model = self.model.cuda().eval()
        self.layers = [ 
                #["slow_pathway_1", 2, self.model.blocks[1].multipathway_blocks[0].res_blocks[-1]],
                ["layer0", 0, self.model.maxpool],
                #["slow_pathway_2", 2, self.model.blocks[2].multipathway_blocks[0].res_blocks[-1]],
                #["layer1", 0, self.model.layer1],
                #["layer2", 0, self.model.layer2],
                ["layer3", 0, self.model.layer3],
                ["layer4", 0, self.model.layer4],
                ["avgpool",0,self.model.avgpool],
                ["fc", 0, self.model.fc],

        ]
        self.batch_model = True
        self.layer_names = [layer[0] for layer in self.layers]
        
    def preprocess_algonauts_video(self, video_path):
        side_size = 256
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        crop_size = 224
        max_frames = 16
        transform =  ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(max_frames),
                    Lambda(lambda x: x/255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(
                        size=side_size
                    ),
                    CenterCropVideo(crop_size)
                ]
            ),
        )
        start_sec = 0
        video = {'video': vid_utils.read_mp4_video(video_path)}
        video_data = transform(video)
        inputs = video_data["video"]
        inputs = inputs.permute(1,0,2,3).cuda()
        return inputs
    

models_dict = {
    "slowfast":SlowfastWrapper,
    "resnet50":ResNet50Wrapper,

}

def get_activations(wrapped_model, video_path, flatten = True):
    activations_dir = {}
    def get_activation(name):
        def hook(model, input, output):
            activations_dir[name] = output.detach().cpu().numpy()
        return hook
    
    handles = []
    for layer in wrapped_model.layers:
        handle = layer[-1].register_forward_hook(get_activation(layer[0]))
        handles.append(handle)
    inputs = wrapped_model.preprocess_algonauts_video(video_path)
    _ = wrapped_model.model(inputs)

    for layer in wrapped_model.layers:
        if type(layer[1]) == list or layer[1] >= 0:
            activations_dir[layer[0]] =activations_dir[layer[0]].mean(axis=layer[1])
        if flatten:
            activations_dir[layer[0]]=activations_dir[layer[0]].flatten()
                
    for handle in handles:
        handle.remove()
    return activations_dir

def get_model_names():
    return list(models_dict.keys())
    
def select_model(model_name):

    return models_dict[model_name]
    
def get_all_activations_and_save(wrapped_model, video_list, activations_folder):
    for video_file in tqdm(video_list):
        video_file_name = os.path.split(video_file)[-1].split(".")[0]
        activations = get_activations(wrapped_model, video_file, True)
    
        for layer in list(activations.keys()):
            save_path = os.path.join(activations_folder, video_file_name+"_"+"layer" + "_" + str(layer) + ".npy")
            np.save(save_path,activations[layer])
    
    