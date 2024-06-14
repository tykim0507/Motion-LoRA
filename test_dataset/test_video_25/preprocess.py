import decord
import cv2
import numpy as np
from diffusers.utils import export_to_video

# Load the video
video = decord.VideoReader('test_dataset/test_video_25/mixkit-highway-in-the-middle-of-a-mountain-range-4633-hd-ready.mp4')
video_frames = video.get_batch(range(0, 25)).asnumpy()
# Convert to numpy array and resize frames
resized_frames = []
for i, frame in enumerate(video_frames):
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    resized_frame = cv2.resize(frame_bgr, (512, 512))
    if i == 0:
        cv2.imwrite("test_dataset/test_image_25/first_frame.png", resized_frame)
    #convert back to rgb
    resized_frame_rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    resized_frames.append(resized_frame_rgb)

output_video_path = "test_dataset/test_video_25/resized_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
h, w, c = resized_frames[0].shape
video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=10, frameSize=(w, h))
for i in range(len(resized_frames)):
    img = cv2.cvtColor(resized_frames[i], cv2.COLOR_RGB2BGR)
    video_writer.write(img)
# export_to_video(resized_frames, "test_dataset/test_video_25/resized_video.mp4")
print("Video saved successfully.")
