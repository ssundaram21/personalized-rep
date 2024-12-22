#!/bin/bash

python generation/generate_data.py \
    --model_path='./finetuned_models/dreambooth_stable-diffusion-v1-5_stable-diffusion-v1-5_2e-06_400' \
    --class_name='mugs_blue_greek' \
    --class_category='mug' \
    --output_path='./synthetic_data/pods' \
    --guidance='5.0' \
    --batch_size='1' \
    --n='100' \
    --inf_steps='50' \
    --prompts_path='./configs/prompts/gpt_prompts_pods.json'
