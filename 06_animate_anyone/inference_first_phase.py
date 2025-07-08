import torch
from animate_anyone.stable_diffusion.ddpm import ImageEmbeddingConditionedLatentDiffusion
from animate_anyone.models.animate_anyone import AnimateAnyone
from omegaconf import OmegaConf
from torch import autocast
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange, repeat
import gc
import random
import numpy as np
from animate_anyone.data.tiktok_dataset import TikTokDataset_SecondPhase
from animate_anyone.data.ubc_fashion_dataset import UBCFashionDataset_SecondPhase
import imageio

CKPT_PATH = "/workspace/jhihyangwu/MyAnimateAnyone/training_runs/00014/model_step_300000.pth"
SD_CKPT_PATH = "/workspace/jhihyangwu/stablediffusion/checkpoints/sd21-unclip-h.ckpt"
SD_CONFIG_PATH = "/workspace/jhihyangwu/MyAnimateAnyone/animate_anyone/stable_diffusion/v2-1-stable-unclip-h-inference.yaml"
DATA_PATH = "/workspace/jhihyangwu/data/ubc_fashion"
DEVICE = "cuda:0"  # do not change, use CUDA_VISIBLE_DEVICES instead inside inference.sh
SEED = 0
REFERENCE_SCENE = 5
POSE_SCENE = 10
NUM_STEPS = 25  # number of ddim sampling steps

# for fun, can load custom image for reference
REFERENCE_IMG = ""  # empty str means to just use dataset img

def load_img(path, final_resolution=512):
    from animate_anyone.utils.utils import center_crop
    img = imageio.imread(path)
    # center crop to final_resolution x final_resolution
    img = center_crop(img, final_resolution)
    # first three channels
    img = img[..., :3]
    # normalize
    img = img / 255.0
    img = 2.0 * img - 1.0
    # convert to tensor
    img = torch.from_numpy(img)
    return img

def main():
    # get device
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    # set seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # load model
    # 1. SD model
    config = OmegaConf.load(SD_CONFIG_PATH)
    sd_weights = torch.load(SD_CKPT_PATH, map_location="cpu", weights_only=False)
    sd_weights = sd_weights["state_dict"]
    # delete useless weights
    for key in ["model_ema.decay", "model_ema.num_updates"]:
        del sd_weights[key]
    sd_model = ImageEmbeddingConditionedLatentDiffusion(**config.model.get("params", dict()))
    sd_model.load_state_dict(sd_weights, strict=True)
    sd_model = sd_model.to(device)
    with autocast(DEVICE):
        empty_c = sd_model.get_learned_conditioning([""])  # CLIP embedding for empty prompt
    # 2. animate anyone model
    model = AnimateAnyone(sd_model, config, device, phase=1).to(device)
    model.eval().requires_grad_(False)
    # 3. copy over trained weights
    state_dict = torch.load(CKPT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    # 4. delete sd model
    del sd_model
    gc.collect()
    torch.cuda.empty_cache()

    # load data
    dataset = UBCFashionDataset_SecondPhase(DATA_PATH, distr="test", inference=True)
    reference_data = dataset[REFERENCE_SCENE]
    pose_data = dataset[POSE_SCENE]

    # get custom reference image if supplied
    if REFERENCE_IMG != "":
        reference_data["expected_images"][0] = load_img(REFERENCE_IMG)

    # create sampler
    num_ddpm_timesteps = config.model.params.timesteps  # should be 1000
    ddim_timesteps = np.array(list(range(0, num_ddpm_timesteps, num_ddpm_timesteps // NUM_STEPS)) + [num_ddpm_timesteps-1])
    linear_start = config.model.params.linear_start
    linear_end = config.model.params.linear_end
    betas = torch.linspace(linear_start ** 0.5, linear_end ** 0.5, num_ddpm_timesteps, dtype=torch.float64) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod = alphas_cumprod[ddim_timesteps]  # only keep alphas used in ddim_timesteps

    with torch.no_grad():
        # get data
        reference_image = reference_data["expected_images"][0].to(torch.float32).to(device).unsqueeze(0)  # (1, H, W, 3)
        pose_sequence = pose_data["expected_poses"].to(torch.float32).to(device)  # (24, H, W, 3)
        orig_video = pose_data["expected_images"].to(torch.float32).to(device)  # (24, H, W, 3)

        reference_image = torch.permute(reference_image, (0, 3, 1, 2))  # (1, C, H, W)
        pose_sequence = torch.permute(pose_sequence, (0, 3, 1, 2))  # (24, C, H, W)
        batch_size = 1
        time_frames = pose_sequence.shape[0]

        # get CLIP features
        adm_cond = model.clip_image_encoder(reference_image)
        weight = 1
        noise_level = 0
        c_adm, noise_level_emb = model.clip_noise_augmentor(adm_cond, noise_level=repeat(
                        torch.tensor([noise_level]).to(reference_image.device), '1 -> b', b=1))
        adm_cond = torch.cat((c_adm, noise_level_emb), 1) * weight
        clip_features = {"c_crossattn": torch.cat([empty_c], 1), "c_adm": adm_cond}

        # convert reference_image into latent space
        reference_latent = model.vae.encoder(reference_image)
        reference_latent = model._sample_latent(reference_latent, deterministic=True)
        reference_latent = model.scale_factor * reference_latent

        # get reference net features
        reference_features = model.reference_net(reference_latent, clip_features)

        # pure noise
        noise = torch.randn(1, 4, dataset.resolution//8, dataset.resolution//8).to(device)  # use the same noise for all time frames

        # denoise each step NUM_STEPS times
        gen_images = []
        for x in trange(time_frames):
            # get pose guidance
            pose_guidance_i = model.pose_guider(pose_sequence[x:x+1])

            t = len(alphas_cumprod) - 1
            alphas_cumprod_t = alphas_cumprod[t]
            x_dec = torch.sqrt(1.0 - alphas_cumprod_t) * noise

            while t >= 0:
                v = model.denoising_unet(x_dec + pose_guidance_i, clip_features, reference_features.copy(), torch.Tensor([ddim_timesteps[t]]).to(device),
                                         batch_size, time_frames, phase=1)
                v -= pose_guidance_i
                # convert v to pred noise / epsilon
                e_t = torch.sqrt(alphas_cumprod_t) * v + torch.sqrt(1.0 - alphas_cumprod_t) * x_dec
                pred_x0 = torch.sqrt(alphas_cumprod_t) * x_dec - torch.sqrt(1.0 - alphas_cumprod_t) * v

                # next x_dec
                if t == 0:
                    alphas_cumprod_t = torch.tensor(1.0, device=x_dec.device)
                else:
                    alphas_cumprod_t = alphas_cumprod[t - 1]
                sigma = 0  # deterministic DDIM sampling
                x_dec = torch.sqrt(alphas_cumprod_t) * pred_x0 + torch.sqrt(1.0 - alphas_cumprod_t - sigma ** 2) * e_t + sigma * torch.randn_like(pred_x0)
                t -= 1
            
            # decode it
            pred_latent = pred_x0
            pred_latent = 1.0 / model.scale_factor * pred_latent
            pred_latent = model.vae.post_quant_conv(pred_latent)

            gen_image = model.vae.decoder(pred_latent)[0]
            gen_image = torch.permute(gen_image, (1, 2, 0))  # (H, W, C)
            gen_images.append(gen_image)
            
        # done for all time frames
        gen_images = torch.stack(gen_images, dim=0)  # (24, H, W, C)
        # save video
        # cols: reference_image, pose_images, gen_images, orig_video
        reference_image = torch.permute(reference_image, (0, 2, 3, 1))  # (1, H, W, C)
        reference_image = repeat(reference_image, "1 h w c -> t h w c", t=time_frames)
        pose_images = torch.permute(pose_sequence, (0, 2, 3, 1))  # (24, H, W, C)
        # to cpu numpy
        reference_image = reference_image.cpu().numpy()
        pose_images = pose_images.cpu().numpy()
        gen_images = gen_images.cpu().numpy()
        orig_video = orig_video.cpu().numpy()
        # combine
        video = np.concatenate([reference_image, pose_images, gen_images, orig_video], axis=2)
        video = np.clip((video + 1.0) / 2.0, 0.0, 1.0)
        video = (video * 255.0).astype(np.uint8)
        imageio.mimwrite("output.mp4", video, fps=24, quality=8)
        print("Video saved to output.mp4")

if __name__ == "__main__":
    main()
