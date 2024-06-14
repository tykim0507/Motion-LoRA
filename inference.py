from svd.pipelines import StableVideoDiffusionPipeline
from diffusers.utils import export_to_gif
import torch
from PIL import Image
#gpu 1
device = "cuda:1"
image_path = "dataset/validation_images/mixkit-flying-over-a-misty-mountain-44616_000_0.jpg"
image = Image.open(image_path)
pipeline = StableVideoDiffusionPipeline.from_pretrained(
            "../models/stable-video-diffusion-img2vid-xt-1-1",
            torch_dtype=torch.float16,
        ).to(device)
video_frames = pipeline(
                image,
                height=512,
                width=512,
                num_frames=21,
            ).frames[0]

export_to_gif(video_frames, "output.gif")


