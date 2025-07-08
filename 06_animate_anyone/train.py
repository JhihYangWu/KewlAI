import click
import torch
import numpy as np
import random
import os
from tqdm import tqdm
from animate_anyone.data.tiktok_dataset import TikTokDataset_FirstPhase, TikTokDataset_SecondPhase
from animate_anyone.data.ubc_fashion_dataset import UBCFashionDataset_FirstPhase, UBCFashionDataset_SecondPhase
from animate_anyone.models.animate_anyone import AnimateAnyone
from animate_anyone.stable_diffusion.ddpm import ImageEmbeddingConditionedLatentDiffusion
from animate_anyone.utils.utils import InfiniteSampler
from animate_anyone.utils import dist
from omegaconf import OmegaConf
import gc
import wandb
from animate_anyone.utils.utils import img_tensor_to_npy
from torch.amp import autocast

@click.command()

# Main options.
@click.option('--outdir',        help='Where to save the results', metavar='DIR',                   type=str, required=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)

# Hyperparameters.
@click.option('--duration',      help='Training duration in steps', metavar='INT',                  type=int, required=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--lr',            help='Learning rate', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=3e-4, show_default=True)

# Performance-related.
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--log_freq',      help='Evaluating logging frequency', metavar='INT',                type=click.IntRange(min=1), default=50, show_default=True)
@click.option('--save_freq',     help='Model saving frequency', metavar='INT',                      type=click.IntRange(min=1), default=50000, show_default=True)

# I/O-related.
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int)
@click.option('--transfer',      help='Transfer weights from a checkpoint', metavar='PTH',          type=str)

# AnimateAnyone-related.
@click.option('--phase',         help='Phase of the training', metavar='INT',                       type=click.IntRange(min=1, max=2), required=True)
@click.option('--sd_config',     help='Path to the stable diffusion config file', metavar='YAML',   type=str, required=True)
@click.option('--sd_ckpt',       help='Path to the stable diffusion checkpoint', metavar='PTH',     type=str, required=True)

def main(**kwargs):
    # parallel training
    try:
        torch.multiprocessing.set_start_method("spawn")
    except:
        pass
    dist.init()

    # set seed
    if kwargs["seed"] is not None:
        seed = kwargs["seed"] + dist.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    # load unmodified stable diffusion
    device = torch.device("cuda")
    config = OmegaConf.load(kwargs["sd_config"])
    sd_ckpt = kwargs["sd_ckpt"]
    print(f"Loading unmodified stable diffusion model from {sd_ckpt}")
    sd_weights = torch.load(sd_ckpt, map_location="cpu", weights_only=False)
    sd_weights = sd_weights["state_dict"]
    # delete useless weights
    for key in ["model_ema.decay", "model_ema.num_updates"]:
        del sd_weights[key]
    sd_model = ImageEmbeddingConditionedLatentDiffusion(**config.model.get("params", dict()))
    sd_model.load_state_dict(sd_weights, strict=True)
    sd_model = sd_model.to(device)

    # construct model
    net = AnimateAnyone(sd_model, config, device, kwargs["phase"])
    net.train().requires_grad_(True)
    empty_c = sd_model.get_learned_conditioning([""])  # CLIP embedding for empty prompt
    del sd_model  # save memory
    gc.collect()
    torch.cuda.empty_cache()

    # transfer over weights from another model
    if kwargs["transfer"] is not None:
        # Load the model
        transfer_model = torch.load(kwargs["transfer"], map_location="cpu", weights_only=True)
        net.load_state_dict(transfer_model, strict=False)
        print(f"Transferred weights from {kwargs['transfer']}")
        del transfer_model  # save memory
        gc.collect()
        torch.cuda.empty_cache()
    
    # create output dir
    if dist.get_rank() == 0:
        os.makedirs(kwargs["outdir"], exist_ok=True)
        prev_run_ids = sorted([int(x) for x in os.listdir(kwargs["outdir"]) if x.isdigit()])
        run_id = f"{prev_run_ids[-1]+1:05}" if prev_run_ids else "00000"
        os.mkdir(os.path.join(kwargs["outdir"], run_id))
        with open(os.path.join(kwargs["outdir"], run_id, "settings.txt"), "w") as f:
            for k, v in kwargs.items():
                f.write(f'{k} = {v}\n')
    
    # load dataset
    phase = kwargs["phase"]
    if phase == 1:
        dataset = UBCFashionDataset_FirstPhase(kwargs["data"])
    elif phase == 2:
        dataset = UBCFashionDataset_SecondPhase(kwargs["data"])
    dataset_sampler = InfiniteSampler(dataset=dataset, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset, sampler=dataset_sampler, batch_size=kwargs["batch"], num_workers=kwargs["workers"], pin_memory=True, drop_last=True))

    # freeze components based on training phase
    # "The weights of the VAE’s Encoder and Decoder, as well as the CLIP image encoder, are all kept fixed"
    net.vae.requires_grad_(False)
    net.clip_image_encoder.requires_grad_(False)
    net.clip_noise_augmentor.requires_grad_(False)
    if phase == 1:
        trainable_params = [p for name, p in net.named_parameters() if
                            p.requires_grad and
                            "vae." not in name and
                            "clip_image_encoder." not in name and
                            "clip_noise_augmentor." not in name and
                            "denoising_unet.temporal_layers." not in name]  # never train the vae or clip_image_encoder or temporal_layers
    elif phase == 2:
        # phase 2 only trains temporal layer, but you need to activate entire denoising_unet because gradients pass through them
        net.requires_grad_(False)
        net.denoising_unet.requires_grad_(True)
        trainable_params = [p for name, p in net.named_parameters() if
                            p.requires_grad and
                            "denoising_unet.temporal_layers." in name]

    # setup ddp
    net = net.to(device)
    ddp = torch.nn.parallel.DistributedDataParallel(net, device_ids=[device], find_unused_parameters=True)

    # optimizer
    optimizer = torch.optim.Adam(trainable_params, lr=kwargs["lr"])

    # setup wandb
    if dist.get_rank() == 0:
        wandb.init(project="MyAnimateAnyone", entity="jhihyang_wu")
        wandb.watch(net, log="parameters", log_freq=kwargs["log_freq"])

    # training loop
    for i in tqdm(range(kwargs["duration"])):
        data = next(dataset_iterator)

        # zero gradients
        optimizer.zero_grad(set_to_none=True)

        # generate noise and timesteps to train on
        time_frames = 1 if phase == 1 else 4
        noise = torch.randn(kwargs["batch"], 4, dataset.resolution//8, dataset.resolution//8).to(torch.float32).to(device)
        noise = noise.repeat_interleave(time_frames, dim=0)  # so for videos, entire video has the same noise
        timesteps = torch.randint(0, config.model.params.timesteps, (kwargs["batch"],), device=device)
        timesteps = timesteps.repeat_interleave(time_frames)  # so for videos, entire video has the same timestep
        timesteps = timesteps.long()

        # forward pass
        ref_images = data["ref_images"].to(torch.float32).to(device)
        expected_images = data["expected_images"].to(torch.float32).to(device)
        expected_poses = data["expected_poses"].to(torch.float32).to(device)
        v, expected_v, pred_x0 = ddp(ref_images, expected_poses, noise, timesteps, empty_c, expected_images, phase)

        # compute loss        
        loss = (v - expected_v).pow(2)

        # backward pass / compute gradients
        loss.mean().backward()

        # step optimizer
        optimizer.step()

        # logging
        if i % kwargs["log_freq"] == 0 and dist.get_rank() == 0:
            print(f"Step {i}: loss={loss.mean().item()}")
            # wandb
            to_log = {}
            to_log["reference_image"] = wandb.Image(img_tensor_to_npy(ref_images[0, 0]))
            to_log["pose_image"] = wandb.Image(img_tensor_to_npy(expected_poses[0, 0]))
            to_log["denoised_image"] = wandb.Image(img_tensor_to_npy(pred_x0))
            to_log["ground_truth"] = wandb.Image(img_tensor_to_npy(expected_images[0, 0]))
            to_log["loss_mean"] = loss.mean()
            to_log["loss_std"] = loss.std()
            to_log["denoised_image_t"] = timesteps[0]
            wandb.log(to_log)

        # save ckpt
        if i % kwargs["save_freq"] == 0 and dist.get_rank() == 0:
            torch.save(net.state_dict(), os.path.join(kwargs["outdir"], run_id, f"model_step_{i}.pth"))
            print(f"Saved model at step {i}")

if __name__ == "__main__":
    main()
