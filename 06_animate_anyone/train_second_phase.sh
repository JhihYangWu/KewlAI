#!/bin/sh

CUDA_VISIBLE_DEVICES=7 torchrun --standalone \
          --nproc_per_node=1 train.py --outdir=training_runs \
                                      --data=/workspace/jhihyangwu/data/ubc_fashion/ \
                                      --lr=1e-5 \
                                      --batch=1 \
                                      --duration=100000000 \
                                      --log_freq=1000 \
                                      --seed=12345 \
                                      --phase=2 \
                                      --transfer=/workspace/jhihyangwu/MyAnimateAnyone/training_runs/00014/model_step_300000.pth \
                                      --sd_config=/workspace/jhihyangwu/MyAnimateAnyone/animate_anyone/stable_diffusion/v2-1-stable-unclip-h-inference.yaml \
                                      --sd_ckpt=/workspace/jhihyangwu/stablediffusion/checkpoints/sd21-unclip-h.ckpt \
