#!/bin/bash

python generation/generate_data.py \
    --model_path='./finetuned_models/dreambooth_masked_stable-diffusion-v1-5_stable-diffusion-v1-5_2e-06_400' \
    --class_name='mugs_blue_greek' \
    --class_category='mug' \
    --output_path='./synthetic_data' \
    --guidance='5.0' \
    --batch_size='1' \
    --n='100' \
    --inf_steps='50' \
    --prompts_path='./configs/prompts/gpt_prompts_pods.json' \
    --filter_thresh='0.6' \
    --train_mask_folder='./data/pods/train_masks' \
    --train_img_folder='./data/pods/train' \
    --mask_ext='jpg' \
    --filter
