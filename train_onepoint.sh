export WANDB_KEY="f2b81eab0686ae96a304a95211b4afa7d5722925"
export PROJECT="SVD-text-control"
export RUN_NAME="one-point"

accelerate launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    train_svd.py \
    --pretrained_model_name_or_path=../models/stable-video-diffusion-img2vid-xt-1-1 \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=6000 \
    --width=512 \
    --height=512 \
    --checkpointing_steps=1000 --checkpoints_total_limit=2 \
    --learning_rate=1e-5 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="bf16" \
    --dataset_path="test_dataset/test_video_25" \
    --validation_steps=200 \
    --num_validation_images=1 \
    --validation_image_path="test_dataset/test_image_25" \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --num_frames=25 \
    --output_dir='./SVD_forward_onepoint' \
    --num_samples=1 \
    --report_to "wandb" 