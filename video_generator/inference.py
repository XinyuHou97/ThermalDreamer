# Copyright 2023 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from model.video_diffusion.models.controlnet3d import ControlNet3DModel
from model.video_diffusion.models.unet_3d_condition import UNetPseudo3DConditionModel
from model.video_diffusion.pipelines.pipeline_stable_diffusion_controlnet3d import Controlnet3DStableDiffusionPipeline
from transformers import DPTForDepthEstimation
from model.annotator.hed import HEDNetwork
import torch
import os
from einops import rearrange,repeat
import imageio
import numpy as np
import cv2
import torch.nn.functional as F
from PIL import Image
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--control_mode', type=str, default='depth', help='support: hed, canny, depth')
parser.add_argument('--inference_step', type=int, default=20, help='denoising steps for inference')
parser.add_argument('--guidance_scale', type=float, default=7.5, help='guidance scale')
parser.add_argument('--seed',  type=int, default=1, help='seed')
parser.add_argument('--num_sample_frames', type=int, default=26, help='total frames to inference')
parser.add_argument('--each_sample_frame', type=int, default=10, help='auto-regressive generation for each iteration')
parser.add_argument('--sampling_rate', type=int, default=1, help='skip sampling from input video')
parser.add_argument('--height', type=int, default=512, help='ouput height')
parser.add_argument('--width', type=int, default=512, help='ouput width')
parser.add_argument('--video_scale', type=float, default=1.5, help='video smoothness scale')
parser.add_argument('--init_noise_thres', type=float, default=0.1, help='thres for res noise init')
# parser.add_argument('--input_video',type=str, default='bear.mp4')
parser.add_argument('--prompt',type=str, default="a bear walking through stars, artstation")
parser.add_argument('--depth_video',type=str, default='')
parser.add_argument('--start', type=int, default=0, help='ouput width')
parser.add_argument('--end', type=int, default=9, help='ouput width')

args = parser.parse_args()

control_mode = args.control_mode
num_inference_steps = args.inference_step
guidance_scale = args.guidance_scale
seed = args.seed
# num_sample_frames = args.num_sample_frames
num_sample_frames = args.end-args.start+1
sampling_rate = args.sampling_rate
h, w = args.height, args.width
video_scale = args.video_scale
init_noise_thres = args.init_noise_thres
# video_path = args.input_video
testing_prompt = [args.prompt]
nega_prompt=["longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, deformed body, bloated, ugly, unrealistic"]
# nega_prompt=[""]
each_sample_frame = args.each_sample_frame


control_net_path = f"weifeng-chen/controlavideo-{control_mode}"
unet = UNetPseudo3DConditionModel.from_pretrained(control_net_path,
                        torch_dtype = torch.float16,
                        subfolder='unet',
                        ).to("cuda") 
controlnet = ControlNet3DModel.from_pretrained(control_net_path,
                        torch_dtype = torch.float16,
                        subfolder='controlnet',
                        ).to("cuda")

if control_mode == 'depth':
    annotator_model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
elif control_mode == 'canny':
    annotator_model = None
elif control_mode == 'hed':
    # firstly download from https://huggingface.co/wf-genius/controlavideo-hed/resolve/main/hed-network.pth 
    annotator_model = HEDNetwork('hed-network.pth').to("cuda")

video_controlnet_pipe = Controlnet3DStableDiffusionPipeline.from_pretrained(control_net_path, unet=unet, 
                        controlnet=controlnet, annotator_model=annotator_model,
                        torch_dtype = torch.float16,
                        ).to("cuda")

# pipeline_model_path = 'stable-diffusion-v1-5'
# state_dict_path = os.path.join(pipeline_model_path, "unet", "diffusion_pytorch_model.bin")
# state_dict = torch.load(state_dict_path, map_location="cpu")
# video_controlnet_pipe.unet.load_2d_state_dict(state_dict=state_dict)    # reload 2d model.


# get video
# np_frames, fps_vid = Controlnet3DStableDiffusionPipeline.get_frames_preprocess(video_path, num_frames=num_sample_frames, sampling_rate=sampling_rate, return_np=True)
if control_mode == 'depth':
    # npyfile= '/mnt/nas/T2R_V/' + args.depth_video + '/depthcrafter/newshin/'+ args.depth_video +'_newshin_depth.npy'
    npyfile= '/mnt/nas/T2R_V/' + args.depth_video + '/depthcrafter/newshin/'+ args.depth_video +'_newshin_depth.npy'

    print(npyfile)
    if args.depth_video:
        depths=np.load(npyfile)
        # print(depths.shape())
        depths=depths[args.start:args.end+1:args.sampling_rate]
        
        # depths=(depths.max()-depths)/(depths.max()-depths.min())
        # print(depths.shape())
        # depths=depths[::2]
        resized_slices = []
        for i in range(depths.shape[0]):
            resized_slice = cv2.resize(depths[i], (512, 512), interpolation=cv2.INTER_NEAREST)
            resized_slices.append(resized_slice)
        control_maps = torch.tensor(np.stack(resized_slices), device="cuda", dtype=torch.float32)
        control_maps = control_maps.unsqueeze(1)
        print(control_maps.shape)
    # else:
    #     frames = torch.from_numpy(np_frames).div(255) * 2 - 1
    #     frames = rearrange(frames, "f h w c -> c f h w").unsqueeze(0)
    #     frames = rearrange(frames, 'b c f h w -> (b f) c h w')
    #     control_maps = video_controlnet_pipe.get_depth_map(frames, h, w, return_standard_norm=False)  # (b f) 1 h w
    
elif control_mode == 'canny':
    # npfile = '/mnt/nas/T2R_gen/depth/'+args.depth_video +'.npy'
    npyfile = '/mnt/nas/T2R_V/'+args.depth_video+'/newshin/'
    # npfile = '/mnt/nas/T2R_gen/depth/'+args.depth_video +'.npy'
    # files=sorted(os.listdir(npfile))[args.start:args.end+1:args.sampling_rate]
    files=np.load(npfile)
    # print(files)
    frames=[]
    for file in files:
        file=(file*255).astype(np.uint8)
        # path= npyfile+file
        # frame = np.array(Image.open(path))
        # frame = Image.open(path)
        # frame = cv2.imread(path)
        frame = cv2.medianBlur(file, 9)
        frame = cv2.Canny(frame, 50, 100)
        frames.append(frame)
    
    # frames=[]
    # for file in files:
    #     path= npyfile+file
    #     frame = np.array(Image.open(path))
    #     # frame = Image.open(path)
    #     # frame = cv2.imread(path)
    #     frame = cv2.medianBlur(frame, 9)
    #     frame = cv2.Canny(frame, 50, 100)
    #     frames.append(frame)
      
    control_maps = np.stack(frames)
    control_maps = repeat(control_maps, 'f h w -> f c h w',c=1)
    control_maps = torch.from_numpy(control_maps).div(255)  # 0~1






# elif control_mode == 'hed':
#     control_maps = np.stack([video_controlnet_pipe.get_hed_map(inp) for inp in np_frames])
#     control_maps = repeat(control_maps, 'f h w -> f c h w',c=1)
#     control_maps = torch.from_numpy(control_maps).div(255)  # 0~1
control_maps = control_maps.to(dtype=controlnet.dtype, device=controlnet.device)
control_maps = F.interpolate(control_maps, size=(h,w), mode='bilinear', align_corners=False)
control_maps = rearrange(control_maps, "(b f) c h w -> b c f h w", f=num_sample_frames)
if control_maps.shape[1] == 1:
    control_maps = repeat(control_maps, 'b c f h w -> b (n c) f h w',  n=3)

# frames = torch.from_numpy(np_frames).div(255)
# frames = rearrange(frames, 'f h w c -> f c h w')
# v2v_input_frames =  torch.nn.functional.interpolate(
#             frames,
#             size=(h, w),
#             mode="bicubic",
#             antialias=True,
#         ) 
# v2v_input_frames = rearrange(v2v_input_frames, '(b f) c h w -> b c f h w ', f=num_sample_frames)


out = []
for i in range(num_sample_frames//each_sample_frame):
    out1 = video_controlnet_pipe(
            # controlnet_hint= control_maps[:,:,:each_sample_frame,:,:],
            # images= v2v_input_frames[:,:,:each_sample_frame,:,:],
            controlnet_hint=control_maps[:,:,i*each_sample_frame-1:(i+1)*each_sample_frame-1,:,:] if i>0 else control_maps[:,:,:each_sample_frame,:,:],
            # images=v2v_input_frames[:,:,i*each_sample_frame-1:(i+1)*each_sample_frame-1,:,:] if i>0 else v2v_input_frames[:,:,:each_sample_frame,:,:],
            first_frame_output=out[-1] if i>0 else None,
            prompt=testing_prompt,
            negative_prompt=nega_prompt,
            num_inference_steps=num_inference_steps,
            width=w,
            height=h,
            guidance_scale=guidance_scale,
            generator=[torch.Generator(device="cuda").manual_seed(seed)],
            video_scale = video_scale,  # per-frame as negative (>= 1 or set 0)
            init_noise_by_residual_thres = init_noise_thres,    # residual-based init. larger thres ==> more smooth.
            controlnet_conditioning_scale=1.0,
            fix_first_frame=True,
            in_domain=True, # whether to use the video model to generate the first frame.
    )
    out1 = out1.images[0][1:]    # drop the first frame
    # out1 = out1.images[0][0:]    # not drop the first frame

    out.extend(out1)

imageio.mimsave('./test/'+args.depth_video+'.gif', out, fps=10, loop=0)


# imageio.mimsave('/mnt/nas/T2R_gen/controlavideo_canny/'+args.depth_video+'.gif', out, fps=10, loop=0)
# imageio.mimsave(args.depth_video.split('/')[-1].split('.')[0]+'.gif', out, fps=10)
# import IPython
# from IPython.display import Image
# Image(filename='demo.gif')