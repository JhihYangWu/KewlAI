# My Unofficial Implementation of AnimateAnyone

## Disclaimer
My implementation is actually quite inefficient. I can only do training on 4 frames at once on a 48 GB card.  
My implementation deviates from the original implementation described by the paper.

## Acknowledgements
- Re-implementation of https://humanaigc.github.io/animate-anyone/
- Copied code from https://github.com/Stability-AI/stablediffusion
- Copied code from https://github.com/NVlabs/edm
- Copied code from https://github.com/guoyww/animatediff/

## Download SD Stable UnCLIP 2.1
- https://huggingface.co/stabilityai/stable-diffusion-2-1-unclip/blob/main/sd21-unclip-h.ckpt

## Download UBC Fashion Dataset
- https://vision.cs.ubc.ca/datasets/fashion/
- Create script for downloading the individual videos and extract frames from it

## Download TikTok Dataset
- Downloaded https://www.kaggle.com/datasets/yasaminjafarian/tiktokdataset
```bash
curl -L -o tiktokdataset.zip https://www.kaggle.com/api/v1/datasets/download/yasaminjafarian/tiktokdataset
```

## Get Poses of People using DWPose
- https://github.com/IDEA-Research/DWPose
- Install the onnx branch
- Get `dw-ll_ucoco_384.onnx` and `yolox_l.onnx` and put them in `ControlNet-v1-1-nightly/annotator/ckpts`
- Try to run `python dwpose_infer_example.py` inside `ControlNet-v1-1-nightly`
- Copy `animate_anyone/data/animate_anyone_dwpose_all.py` to `ControlNet-v1-1-nightly` and run it

## xformers
You need to install and use xformers or else you won't have enough memory for even a batch size of 2
```
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
pip install -U xformers --index-url https://download.pytorch.org/whl/cu124
```

## First Stage of Training
```bash
./train_first_phase.sh
```
Training run results: [wandb](https://wandb.ai/jhihyang_wu/MyAnimateAnyone/runs/orhogsie?nw=nwuserjhihyang_wu)

## Inference First Stage
```bash
source ./inference_first_phase.sh
```

## Second Stage of Training
```bash
./train_second_phase.sh
```
Training run results: [wandb_1](https://wandb.ai/jhihyang_wu/MyAnimateAnyone/runs/9bg1m5fw?nw=nwuserjhihyang_wu) [wandb_2](https://wandb.ai/jhihyang_wu/MyAnimateAnyone/runs/eyqgcbbj?nw=nwuserjhihyang_wu)

## Inference Second Stage
```
source ./inference_second_phase.sh
```

## Trained Weights
You can download the weights that I trained for ~2 weeks (~1 week per phase) on a single Nvidia L40S here.
