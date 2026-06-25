from types import MethodType
import os

from click import prompt
import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
from PIL import Image
import argparse
from ip_adapter import IPAdapter
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
args = parser.parse_args()

hint='./data/'+args.data+'/scene-latent/scene_latent_ref.png'

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "scene_latent/models/image_encoder/"
ip_ckpt = "scene_latent/models/ip-adapter_sd15.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)


# load controlnet
controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
# load SD pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)


# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
hintimg=Image.open(hint)
depth=np.load('./data/'+args.data+'/depth/'+args.data+'.npz')['depth'][0]



gen_imgs=[]
Image_depth=Image.fromarray((depth*255).astype(np.uint8))
image = ip_model.generate(pil_image=hintimg, prompt=args.prompt, image=Image_depth, num_samples=1, num_inference_steps=50, seed=42)

output_path = "./data/"+args.data+"/scene-latent/firstframe_depth.png"
image[0].save(output_path)



hint=hintimg.resize((512,512),Image.BICUBIC)
Image_depth=Image_depth.resize((512,512),Image.NEAREST)
# print(startimg.size)
image = ip_model.generate(pil_image=hintimg, prompt=args.prompt, image=Image_depth, num_samples=1, num_inference_steps=50, seed=42,output_type = "latent")
# print(image.shape)
output_path = "./data/"+args.data+"/scene-latent/firstframe_depth.npy"
# # gen_imgs[0].save(output_path, save_all=True, append_images=gen_imgs[1:], duration=100, loop=0)
# # image[0].save(output_path)
np.save(output_path,image.cpu().numpy())






