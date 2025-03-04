#!/bin/bash
mkdir -p ./cache
cd cache

## Download pretrained checkpoints
wget -O clip_vitb16_pretrain.pth.tar https://data.csail.mit.edu/personal_rep/checkpoints/clip_vitb16_pretrain.pth.tar
wget -O ViT-B-16.pt https://data.csail.mit.edu/personal_rep/checkpoints/ViT-B-16.pt
