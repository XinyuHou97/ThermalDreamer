import gc
import os
import numpy as np
import torch
import time

from diffusers.training_utils import set_seed
from fire import Fire

from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import vis_sequence_depth, save_video, read_video_frames_from_folder


class DepthCrafterDemo:
    def __init__(
        self,
        unet_path: str,
        pre_train_path: str,
        cpu_offload: str = "model",
    ):
        unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
            unet_path,
            low_cpu_mem_usage=True,
            torch_dtype=torch.float16,
        )
        # load weights of other components from the provided checkpoint
        self.pipe = DepthCrafterPipeline.from_pretrained(
            pre_train_path,
            unet=unet,
            torch_dtype=torch.float16,
            variant="fp16",
        )

        # for saving memory, we can offload the model to CPU, or even run the model sequentially to save more memory
        if cpu_offload is not None:
            if cpu_offload == "sequential":
                # This will slow, but save more memory
                self.pipe.enable_sequential_cpu_offload()
            elif cpu_offload == "model":
                self.pipe.enable_model_cpu_offload()
            else:
                raise ValueError(f"Unknown cpu offload option: {cpu_offload}")
        else:
            self.pipe.to("cuda")
        # enable attention slicing and xformers memory efficient attention
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(e)
            print("Xformers is not enabled")
        self.pipe.enable_attention_slicing()

    def infer(
        self,
        video: str,
        num_denoising_steps: int,
        guidance_scale: float,
        save_folder: str = "./demo_output",
        window_size: int = 110,
        process_length: int = 195,
        overlap: int = 25,
        max_res: int = 1024,
        dataset: str = "open",
        target_fps: int = 15,
        seed: int = 42,
        track_time: bool = True,
        save_npz: bool = False,
        save_exr: bool = False,
    ):
        set_seed(seed)

        frames, target_fps = read_video_frames_from_folder(
            video,
            process_length,
            target_fps,
            max_res,
            dataset,
        )
        # inference the depth map using the DepthCrafter pipeline
        with torch.inference_mode():
            res = self.pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]
        # convert the three-channel output to a single channel depth map
        res = res.sum(-1) / res.shape[-1]
        # normalize the depth map to [0, 1] across the whole video
        res = (res - res.min()) / (res.max() - res.min())
        # visualize the depth map and save the results
        vis = vis_sequence_depth(res)
        # save the depth map and visualization with the target FPS
        save_path = os.path.join(
            save_folder, video.split('/')[2]#os.path.splitext(os.path.basename(video))[0]
        )
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save numpy arrays first (these don't require ffmpeg)
        if save_npz:
            np.savez_compressed(save_path + ".npz", depth=res)
            print(f"Saved depth data to: {save_path}.npz")
        
        # Try to save videos, but continue if ffmpeg is not available
        try:
            save_video(res, save_path + "_depth.mp4", fps=target_fps)
            save_video(vis, save_path + "_vis.mp4", fps=target_fps)
            save_video(frames, save_path + "_input.mp4", fps=target_fps)
            print(f"Saved videos to: {save_path}_*.mp4")
        except RuntimeError as e:
            if "ffmpeg" in str(e):
                print(f"Warning: Could not save videos due to missing ffmpeg: {e}")
                print("Depth data has been saved as .npz file which can be used for further processing.")
            else:
                raise e
        if save_exr:
            import OpenEXR
            import Imath

            os.makedirs(save_path, exist_ok=True)
            print(f"==> saving EXR results to {save_path}")
            # Iterate over each frame and save as a separate EXR file
            for i, frame in enumerate(res):
                output_exr = f"{save_path}/frame_{i:04d}.exr"

                # Prepare EXR header for each frame
                header = OpenEXR.Header(frame.shape[1], frame.shape[0])
                header["channels"] = {
                    "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                }

                # Create EXR file and write the frame
                exr_file = OpenEXR.OutputFile(output_exr, header)
                exr_file.writePixels({"Z": frame.tobytes()})
                exr_file.close()

        return [
            save_path + "_input.mp4",
            save_path + "_vis.mp4",
            save_path + "_depth.mp4",
        ]

    def run(
        self,
        input_video,
        num_denoising_steps,
        guidance_scale,
        max_res=1024,
        process_length=195,
    ):
        res_path = self.infer(
            input_video,
            num_denoising_steps,
            guidance_scale,
            max_res=max_res,
            process_length=process_length,
        )
        # clear the cache for the next video
        gc.collect()
        torch.cuda.empty_cache()
        return res_path[:2]


def main(
    data: str,
    save_folder: str = "./demo_output",
    unet_path: str = "tencent/DepthCrafter",
    pre_train_path: str = "stabilityai/stable-video-diffusion-img2vid-xt",
    process_length: int = -1,
    cpu_offload: str = "model",
    target_fps: int = -1,
    seed: int = 42,
    num_inference_steps: int = 5,
    guidance_scale: float = 1.0,
    window_size: int = 32,
    overlap: int = 8,
    max_res: int = 512,
    dataset: str = "open",
    save_npz: bool = True,
    save_exr: bool = False,
    track_time: bool = False,
):
    # Validate overlap parameter
    if overlap >= window_size:
        print(f"Warning: overlap ({overlap}) should be less than window_size ({window_size}). Adjusting overlap to {window_size - 1}.")
        overlap = window_size - 1
    
    # Check if we can optimize parameters for short videos
    import os
    import glob
    
    data_path = os.path.join("./data", data, "thermal-enh")
    
    # Count frames in the thermal-enh directory
    if os.path.exists(data_path):
        frame_files = glob.glob(os.path.join(data_path, "*.png")) + glob.glob(os.path.join(data_path, "*.jpg"))
        num_frames = len(frame_files)
        
        if num_frames > 0:
            print(f"Detected {num_frames} frames in the video.")
            
            # Optimize parameters for short videos (≤40 frames)
            if num_frames <= 40:
                if num_frames <= window_size:
                    # Video is shorter than window size, process in one go
                    recommended_window = num_frames
                    recommended_overlap = 0
                    print(f"Short video detected. Recommending: window_size={recommended_window}, overlap={recommended_overlap}")
                    window_size = recommended_window
                    overlap = recommended_overlap
                else:
                    # Video needs 2-3 windows
                    recommended_window = min(24, max(16, num_frames // 2 + 4))
                    recommended_overlap = min(8, recommended_window // 3)
                    print(f"Medium video detected. Recommending: window_size={recommended_window}, overlap={recommended_overlap}")
                    window_size = recommended_window
                    overlap = recommended_overlap
            else:
                print(f"Using provided parameters: window_size={window_size}, overlap={overlap}")
    
    depthcrafter_demo = DepthCrafterDemo(
        unet_path=unet_path,
        pre_train_path=pre_train_path,
        cpu_offload=cpu_offload,
    )
    # process the videos, the video paths are separated by comma

    save_folder = os.path.join("./data", data, "depth")
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    start_time = time.perf_counter()
    depthcrafter_demo.infer(
        data_path,
        num_inference_steps,
        guidance_scale,
        save_folder=save_folder,
        window_size=window_size,
        process_length=process_length,
        overlap=overlap,
        max_res=max_res,
        dataset=dataset,
        target_fps=target_fps,
        seed=seed,
        track_time=track_time,
        save_npz=save_npz,
        save_exr=save_exr,
    )
    end_time = time.perf_counter()
    print(f"Total inference time: {end_time - start_time:.2f} seconds")
    # clear the cache for the next video
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # running configs
    # the most important arguments for memory saving are `cpu_offload`, `enable_xformers`, `max_res`, and `window_size`
    # the most important arguments for trade-off between quality and speed are
    # `num_inference_steps`, `guidance_scale`, and `max_res`
    Fire(main)
