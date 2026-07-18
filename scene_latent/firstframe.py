import time
import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler, AutoencoderKL, ControlNetModel
from PIL import Image
import argparse
from ip_adapter import IPAdapter
import numpy as np
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--prompt", type=str, required=True)
parser.add_argument(
    "--control_model",
    "--control_mode",
    dest="control_model",
    type=str,
    default="depth",
    choices=["depth", "canny", "fuse"],
    help="Choose control source: depth map, canny edge, or fuse (treated as depth).",
)
args = parser.parse_args()

effective_control_model = "depth" if args.control_model == "fuse" else args.control_model

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
controlnet_model_path_map = {
    "depth": "lllyasviel/control_v11f1p_sd15_depth",
    "canny": "lllyasviel/control_v11p_sd15_canny",
}
controlnet_model_path = controlnet_model_path_map[effective_control_model]
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


if effective_control_model == "depth":
    control_arr = np.load('./data/'+args.data+'/depth/'+args.data+'.npz')['depth'][0]
    control_image = Image.fromarray((control_arr * 255).astype(np.uint8))
    output_suffix = "depth"
else:
    startimg = Image.open('./data/'+args.data+'/thermal-enh/000000.png')
    frame = np.array(startimg)
    frame = cv2.medianBlur(frame, 9)
    frame = cv2.Canny(frame, 50, 100)
    control_image = Image.fromarray(frame)
    output_suffix = "canny"


gen_imgs=[]
control_image = control_image.resize((512,512),Image.NEAREST)

start = time.perf_counter()
image = ip_model.generate(pil_image=hintimg, prompt=args.prompt, image=control_image, num_samples=1, num_inference_steps=50, seed=42)
end = time.perf_counter()
print(f"Time: {end - start:.4f}s")

output_path = "./data/"+args.data+"/scene-latent/firstframe_" + output_suffix + ".png"
image[0].save(output_path)



hintimg=hintimg.resize((512,512),Image.BICUBIC)
image = ip_model.generate(pil_image=hintimg, prompt=args.prompt, image=control_image, num_samples=1, num_inference_steps=50, seed=42,output_type = "latent")
output_path = "./data/"+args.data+"/scene-latent/firstframe_" + output_suffix + ".npy"
# # gen_imgs[0].save(output_path, save_all=True, append_images=gen_imgs[1:], duration=100, loop=0)
# # image[0].save(output_path)
np.save(output_path,image.cpu().numpy())






