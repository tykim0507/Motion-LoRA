
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_gif
import numpy as np
import torch
from glob import glob
import cv2
from PIL import Image
import imageio
import os


if __name__ == "__main__":
    device = "cuda"
    video_path_list = sorted(glob("dataset/forward/*_000_0.mp4"))
    video_path_list = video_path_list[:50]

    loras = ["forward"]
    ckpt_steps = [4000]
    
    for lora, ckpt_step in zip(loras, ckpt_steps):

        pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "/models/stable-video-diffusion-img2vid-xt-1-1",
            torch_dtype=torch.float16,
        )
        pipeline = pipeline.to(device)
        
        lora_pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "/models/stable-video-diffusion-img2vid-xt-1-1",
            torch_dtype=torch.float16,
        )
        lora_pipeline = lora_pipeline.to(device)
        
        output_dir = f"results_lora/{lora}-{ckpt_step}"
        os.makedirs(output_dir, exist_ok=True)
        
        
        lora_pipeline.load_lora_weights(f"./LoRA_{lora}_all/checkpoint-{ckpt_step}/pytorch_lora_weights.safetensors")
        
        
        

        for video_path in video_path_list:
            video_name = video_path.split("/")[-1].split(".")[0]
            cap = cv2.VideoCapture(video_path)
            _, first_img = cap.read()
            cap.release()

            first_img = cv2.cvtColor(first_img, cv2.COLOR_BGR2RGB)
            first_img = Image.fromarray(first_img)

            first_img.save(os.path.join(output_dir, video_name) + '_first_frame.png')

            video_frames = pipeline(
                first_img,
                height=512,
                width=512,
                num_inference_steps=30,
                max_guidance_scale=3.0,
            ).frames[0]

            export_to_gif(video_frames, os.path.join(output_dir, video_name) + 'naiveSVD.gif', 8)
            
            
            
            video_frames = lora_pipeline(
                first_img,
                height=512,
                width=512,
                num_inference_steps=30,
                max_guidance_scale=3.0,
            ).frames[0]
            
            export_to_gif(video_frames, os.path.join(output_dir, video_name) + '.gif', 8)
            
