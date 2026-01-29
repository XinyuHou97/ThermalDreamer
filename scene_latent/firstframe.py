from types import MethodType
import os
import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
from PIL import Image

from ip_adapter import IPAdapter
import numpy as np
base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/image_encoder/"
ip_ckpt = "models/ip-adapter_sd15.bin"
device = "cuda"



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

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

# read image prompt
# image = Image.open("assets/images/statue.png")
# depth_map = Image.open("assets/structure_controls/depth.png")
# image_grid([image.resize((256, 256)), depth_map.resize((256, 256))], 1, 2)


# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)


scene_prompts = {
    "2atrium": "a modern office hallway area featuring a set of metal chairs and a table",
    "atrium": "a quiet office or library space with black sofas",
    "canteen": "a brightly lit hallway with blue doors",
    "corridor": "a well-lit corridor with lockers",
    "indoor_robust_dark": "a dark indoor area with a table and chairs",
    "indoor_robust_global": "a dark indoor area with wide-angle view, a table and some chairs",
    "office": "an office with blue carpeting, featuring a row of beige lockers on the right,",
    "outdoor_robust_day": "an bright outdoor parking lot d near a modern building with parked cars, trees, streetlights",
    "outdoor_robust_night": "an outdoor nighttime scene featuring a building, trees, streetlights"
}

# image_grid(ims,5,6)
# image_grid([Image_depth],1,1)


# vid=4
# vid=str(vid)

# startimg=Image.open('assets/obv/'+vid+'start.png')
# depths=np.load('assets/obv/depths/'+vid+'.npz')
# depths=depths['depth']

# image_grid([startimg],1,1)

hints=["/mnt/nas/T2R_V/2atrium1/rgb/0000.png",
"/mnt/nas/T2R_V/atrium0/rgb/0080.png",
"/mnt/nas/T2R_V/atrium1/rgb/0000.png",
"/mnt/nas/T2R_V/atrium1/rgb/0000.png",
"/mnt/nas/T2R_V/canteen0/rgb/0000.png",
"/mnt/nas/T2R_V/corridor0/rgb/0092.png",
# "/hdd2/xinyu/ThermalMonoDepth/data/indoor_robust_global/RGB/data/000782.png",
# "/hdd2/xinyu/ThermalMonoDepth/data/indoor_robust_global/RGB/data/000782.png",
# "/hdd2/xinyu/ThermalMonoDepth/data/indoor_robust_global/RGB/data/000782.png",
# "/hdd2/xinyu/ThermalMonoDepth/data/indoor_robust_global/RGB/data/000371.png",
# "/hdd2/xinyu/ThermalMonoDepth/data/indoor_robust_global/RGB/data/000371.png",
# "/hdd2/xinyu/ThermalMonoDepth/data/indoor_robust_global/RGB/data/000371.png",
"/mnt/nas/T2R_V/atrium0/rgb/0000.png",
"/mnt/nas/T2R_V/atrium0/rgb/0000.png",
"/mnt/nas/T2R_V/atrium0/rgb/0000.png",
# "/mnt/nas/T2R_V/000837.png",
# "/mnt/nas/T2R_V/000837.png",
# "/mnt/nas/T2R_V/000837.png",
# "/mnt/nas/T2R_V/2atrium1/rgb/0000.png",
# "/mnt/nas/T2R_V/2atrium1/rgb/0000.png",
# "/mnt/nas/T2R_V/2atrium1/rgb/0000.png",
"/mnt/nas/T2R_V/office0/rgb/0099.png",
"/mnt/nas/T2R_V/office0/rgb/0099.png",
"/mnt/nas/T2R_V/office0/rgb/0099.png",
"/mnt/nas/T2R_V/office0/rgb/0000.png",
"/mnt/nas/T2R_V/office0/rgb/0000.png",
"/mnt/nas/T2R_V/outdoor_robust_day10/rgb/000100.png",
"/mnt/nas/T2R_V/outdoor_robust_day10/rgb/000001.png",
"/mnt/nas/T2R_V/outdoor_robust_day11/rgb/000770.png",
"/mnt/nas/T2R_V/outdoor_robust_day11/rgb/000770.png",
"/mnt/nas/T2R_V/outdoor_robust_day11/rgb/000770.png",
"/mnt/nas/T2R_V/outdoor_robust_day11/rgb/000770.png"]





folder='/mnt/nas/T2R_gen/depth/'

seqs=sorted(os.listdir(folder))

print(seqs)

# seq='2atrium0_65''/mnt/nas/T2R_gen/depth/'

for i, seq in enumerate(seqs):
    # startimg=Image.open('/mnt/nas/T2R_gen/IP_adaptor/'+seq+'_start.png')
    startimg=Image.open(hints[i])
    depth=np.load('/mnt/nas/T2R_gen/depth/'+seq)[0]
    # depths=depths['depth']
    
    # for pro in scene_prompts:
    #     if seq.startswith(pro):
    #         ppp=scene_prompts[pro]
    #         break
    
    if 'outdoor' in seq:
        prompt='campus'
    else:
        prompt='office room'
    
    gen_imgs=[]
    # for depth in depths:
        # print(depth)
    Image_depth=Image.fromarray((depth*255).astype(np.uint8))
    image = ip_model.generate(pil_image=startimg, prompt=prompt, image=Image_depth, num_samples=1, num_inference_steps=50, seed=42)
    # gen_imgs.append(images[0])
    
    
    # ims=[i[0] for i in gen_imgs]
    
    output_path = "/mnt/nas/T2R_gen/firstframe-short2/"+seq+".png"
    # gen_imgs[0].save(output_path, save_all=True, append_images=gen_imgs[1:], duration=100, loop=0)
    image[0].save(output_path)
    
    
    # startimg=startimg.resize((512,512),Image.BICUBIC)
    # Image_depth=Image_depth.resize((512,512),Image.NEAREST)
    # print(startimg.size)
    # image = ip_model.generate(pil_image=startimg, prompt=prompt, image=Image_depth, num_samples=1, num_inference_steps=50, seed=42,output_type = "latent")
    # print(image.shape)
    # output_path = "/mnt/nas/T2R_gen/firstframe-short/"+seq
    # # gen_imgs[0].save(output_path, save_all=True, append_images=gen_imgs[1:], duration=100, loop=0)
    # # image[0].save(output_path)
    # np.save(output_path,image.cpu().numpy())
        
    
    
    
    
    
    
    
    # images = ip_model.generate(pil_image=image, image=depth_map, num_samples=1, num_inference_steps=50, seed=42)
    # grid = image_grid(images, 1, 1)
    # grid
    
    













