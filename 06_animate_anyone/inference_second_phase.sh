#!/bin/sh

# run with "source ./inference_second_phase.sh"

conda activate animateanyone
CUDA_VISIBLE_DEVICES=5 python inference_second_phase.py
