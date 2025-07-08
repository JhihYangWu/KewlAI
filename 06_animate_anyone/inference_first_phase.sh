#!/bin/sh

# run with "source ./inference_first_phase.sh"

conda activate animateanyone
CUDA_VISIBLE_DEVICES=5 python inference_first_phase.py
