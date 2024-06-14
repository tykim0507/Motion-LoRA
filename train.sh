export PROJECT="svd-finetune-LoRA"
accelerate launch \
    --config_file scripts/accelerate_configs/single_gpu.yaml \
    train_svd.py \
    --pretrained_model_name_or_path=checkpoints/stable-video-diffusion-img2vid-xt-1-1 \
    --per_gpu_batch_size=1 --gradient_accumulation_steps=1 \
    --max_train_steps=6000 \
    --width=512 \
    --height=512 \
    --checkpointing_steps=200 --checkpoints_total_limit=30 \
    --learning_rate=1e-4 --lr_warmup_steps=0 \
    --seed=123 \
    --mixed_precision="bf16" \
    --dataset_path="dataset/forward" \
    --validation_steps=200 \
    --num_validation_images=1 \
    --validation_image_path="dataset/validation_images" \
    --gradient_checkpointing \
    --enable_xformers_memory_efficient_attention \
    --num_frames=21 \
    --output_dir='./LoRA_forward_onepoint' \
    --train_lora \
    --rank=16 \
    --num_samples=1 \
    --report_to "wandb" 