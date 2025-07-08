# written by JhihYang Wu <jhihyangwu@arizona.edu>

import torch.nn as nn
from copy import deepcopy
import torch
from animate_anyone.stable_diffusion.utils import timestep_embedding
import numpy as np
from einops import repeat, rearrange
from animate_anyone.animatediff.animatediff import VanillaTemporalModule

class AnimateAnyone(nn.Module):
    def __init__(self, stable_diffusion, sd_config, device, phase):
        super(AnimateAnyone, self).__init__()
        self.pose_guider = PoseGuider()

        self.scale_factor = stable_diffusion.scale_factor
        self.vae = deepcopy(stable_diffusion.first_stage_model)
        self.clip_image_encoder = deepcopy(stable_diffusion.embedder)
        self.clip_noise_augmentor = deepcopy(stable_diffusion.noise_augmentor)

        sd_unet = stable_diffusion.model.diffusion_model
        self.denoising_unet = DenoisingUNet(deepcopy(sd_unet), phase)
        self.reference_net = ReferenceNet(deepcopy(sd_unet))

        # for forward process
        # from DDPM paper
        betas = torch.linspace(sd_config.model.params.linear_start ** 0.5,
                               sd_config.model.params.linear_end ** 0.5,
                               sd_config.model.params.timesteps,
                               dtype=torch.float64,
                               device=device) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.total_timesteps = sd_config.model.params.timesteps

    def _sample_latent(self, h, deterministic=False):
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)
        std = torch.exp(0.5 * logvar)
        if deterministic:
            return mean
        latent = mean + std * torch.randn(mean.shape).to(device=std.device)
        return latent

    def forward(self, reference_image, pose_sequence, noise, timesteps, empty_c, video_clip, phase):
        # get rid of time dimension when not needed
        assert len(reference_image.shape) == 5 and len(pose_sequence.shape) == 5 and len(video_clip.shape) == 5
        batch_size = video_clip.shape[0]
        time_frames = video_clip.shape[1]
        reference_image = reference_image.view(batch_size, *reference_image.shape[2:])  # (B, H, W, C)
        pose_sequence = pose_sequence.view(batch_size * time_frames, *pose_sequence.shape[2:])  # (B*T, H, W, C)
        video_clip = video_clip.view(batch_size * time_frames, *video_clip.shape[2:])  # (B*T, H, W, C)
        reference_image = torch.permute(reference_image, (0, 3, 1, 2))  # (B, C, H, W)
        pose_sequence = torch.permute(pose_sequence, (0, 3, 1, 2))  # (B*T, C, H, W)
        video_clip = torch.permute(video_clip, (0, 3, 1, 2))  # (B*T, C, H, W)

        # get CLIP features
        with torch.no_grad():
            adm_cond = self.clip_image_encoder(reference_image)
            weight = 1
            noise_level = 0
            c_adm, noise_level_emb = self.clip_noise_augmentor(adm_cond, noise_level=repeat(
                            torch.tensor([noise_level]).to(reference_image.device), '1 -> b', b=1))
            adm_cond = torch.cat((c_adm, noise_level_emb), 1) * weight

            empty_c = empty_c.expand(batch_size, *empty_c.shape[1:])
            adm_cond = adm_cond.expand(batch_size, *adm_cond.shape[1:])
            clip_features = {"c_crossattn": torch.cat([empty_c], 1), "c_adm": adm_cond}

        # convert reference_image and video_clip into latent space
        with torch.no_grad():
            reference_latent = self.vae.encoder(reference_image)
            reference_latent = self._sample_latent(reference_latent)
            reference_latent = self.scale_factor * reference_latent
            video_clip_latent = self.vae.encoder(video_clip)
            video_clip_latent = self._sample_latent(video_clip_latent)
            video_clip_latent = self.scale_factor * video_clip_latent
        
        if phase == 1:
            # get reference net features
            reference_features = self.reference_net(reference_latent, clip_features)
            # get pose guidance
            pose_guidance = self.pose_guider(pose_sequence)
        else:
            assert phase == 2
            with torch.no_grad():
                reference_features = self.reference_net(reference_latent, clip_features)
                pose_guidance = self.pose_guider(pose_sequence)

        # create noisy video clip latent based on timesteps
        # forward process
        alphas_cumprod_t = self.alphas_cumprod[timesteps]
        alphas_cumprod_t = alphas_cumprod_t.view(batch_size * time_frames, 1, 1, 1)
        noised_video_clip_latent = (
            torch.sqrt(alphas_cumprod_t) * video_clip_latent +
            torch.sqrt(1.0 - alphas_cumprod_t) * noise  # sqrt because standard deviation
        ).to(torch.float32)

        # get denoising unet to predict v
        v = self.denoising_unet(noised_video_clip_latent + pose_guidance, clip_features, reference_features, timesteps,
                                batch_size, time_frames, phase)
        v -= pose_guidance
        # convert v to pred noise / epsilon
        e_t = torch.sqrt(alphas_cumprod_t) * v + torch.sqrt(1.0 - alphas_cumprod_t) * noised_video_clip_latent
        pred_x0 = torch.sqrt(alphas_cumprod_t) * noised_video_clip_latent - torch.sqrt(1.0 - alphas_cumprod_t) * v
        e_t = e_t.to(torch.float32)
        pred_x0 = pred_x0.to(torch.float32)

        expected_v = (
            torch.sqrt(alphas_cumprod_t) * noise -
            torch.sqrt(1.0 - alphas_cumprod_t) * video_clip_latent
        ).to(torch.float32)

        # convert pred_x0 to image space
        with torch.no_grad():
            pred_latent = pred_x0
            pred_latent = 1.0 / self.scale_factor * pred_latent
            pred_latent = self.vae.post_quant_conv(pred_latent)

            gen_image = self.vae.decoder(pred_latent)[0]
            gen_image = torch.permute(gen_image, (1, 2, 0))  # (H, W, C)

        return v, expected_v, gen_image

# """
# This Pose Guider utilizes four convolution layers (4×4 kernels, 2×2 strides,
# using 16,32,64,128 channels, similar to
# the condition encoder in [60]) to align the pose image with
# the same resolution as the noise latent. Subsequently, the
# processed pose image is added to the noise latent before being input into the denoising UNet.
# The Pose Guider is initialized with Gaussian weights, and in the final projection
# layer, we employ zero convolution
# """

class PoseGuider(nn.Module):
    def __init__(self):
        super(PoseGuider, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)   # (B, 16, H/2, W/2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)  # (B, 32, H/4, W/4)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # (B, 64, H/8, W/8)
        # self.conv4 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # (B, 128, H/16, W/16)
        # TODO: make stable diffusion downsampling factor to 16 and uncomment above line and mod forward etc.

        self.final_proj = nn.Conv2d(64, 4, kernel_size=1, stride=1, padding=0)  # (B, 4, H/16, W/16)

        self.initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # x = self.conv4(x)
        x = self.final_proj(x)
        return x

    def initialize_weights(self, mean=0.0, std=1.0):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # zero convolution for final projection layer
        nn.init.zeros_(self.final_proj.weight)
        nn.init.zeros_(self.final_proj.bias)

class ReferenceNet(nn.Module):
    def __init__(self, sd_unet):
        super(ReferenceNet, self).__init__()
        self.sd_unet = sd_unet

        # hack forward method of sd_unet to output all intermediate features
        def hack_forward(self, x, timesteps=None, context=None, y=None,**kwargs):
            assert y is not None
            retval = []
            hs = []
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

            h = x.type(self.dtype)
            retval.append(h)
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
                retval.append(h)
            h = self.middle_block(h, emb, context)
            retval.append(h)
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context)
                retval.append(h)
            h = h.type(x.dtype)
            
            return retval
        
        self.sd_unet_hack_forward = hack_forward

    def forward(self, reference_latent, clip_features):
        # no noise should be added to the reference latent
        # so time should be zero
        batch_size = reference_latent.size(0)
        t = torch.zeros(batch_size).to(reference_latent.device)

        reference_features = self.sd_unet_hack_forward(self.sd_unet, reference_latent, t,
                                                       context=clip_features["c_crossattn"], y=clip_features["c_adm"])
        return reference_features

class DenoisingUNet(nn.Module):
    def __init__(self, sd_unet, phase):
        super(DenoisingUNet, self).__init__()
        self.sd_unet = sd_unet

        # hack forward method of sd_unet to add reference_net_feats and temporal attention

        # """
        # Specifically, as shown in Fig. 2, we
        # replace the self-attention layer with spatial-attention layer.
        # Given a feature map x1∈R
        # t×h×w×c
        # from denoising UNet
        # and x2∈R
        # h×w×c
        # from ReferenceNet, we first copy x2 by t
        # times and concatenate it with x1 along w dimension. Then
        # we perform self-attention and extract the first half of the feature map as the output. 
        # """

        def hack_forward(self, x, timesteps=None, context=None, y=None, ref_net_feats=None,**kwargs):
            assert y is not None
            assert ref_net_feats is not None
            hs = []
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

            h = x.type(self.dtype)
            h = torch.cat([h, ref_net_feats.pop(0)], dim=-1)  # ref net features

            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
                h = h[..., :h.shape[-1] // 2]  # first half
                h = torch.cat([h, ref_net_feats.pop(0)], dim=-1)  # ref net features
            h = self.middle_block(h, emb, context)
            h = h[..., :h.shape[-1] // 2]  # first half
            h = torch.cat([h, ref_net_feats.pop(0)], dim=-1)  # ref net features
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context)
                h = h[..., :h.shape[-1] // 2]  # first half
                h = torch.cat([h, ref_net_feats.pop(0)], dim=-1)  # ref net features
            h = h.type(x.dtype)

            h = h[..., :h.shape[-1] // 2]  # first half
            assert len(ref_net_feats) == 0

            return self.out(h)
        
        self.sd_unet_hack_forward = hack_forward

        # ==== phase 2 things ====

        # need some temporal layers
        self.temporal_layers = nn.ModuleList([])
        IN_CHANNELS = [320] * 4 + [640] * 3 + [1280] * 12 + [640] * 3 + [320] * 3
        TEMPORAL_LAYERS_KWARGS = {
            "num_attention_heads":        8,
            "num_transformer_block":      1,
            "attention_block_types":      [ "Temporal_Self", "Temporal_Self" ],
            "temporal_position_encoding": True,
            "temporal_attention_dim_div": 1,
            "zero_initialize":            True,
        }
        if phase == 2:
            for in_channels in IN_CHANNELS:
                self.temporal_layers.append(
                    VanillaTemporalModule(
                        in_channels=in_channels,
                        **TEMPORAL_LAYERS_KWARGS
                    )
                )

        def hack_forward_phase_2(self, x, timesteps=None, context=None, y=None, ref_net_feats=None, temporal_layers=None, batch_size=None, time_frames=None, orig_context=None,**kwargs):
            assert y is not None
            assert ref_net_feats is not None
            hs = []
            t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
            emb = self.time_embed(t_emb)

            assert y.shape[0] == x.shape[0]
            emb = emb + self.label_emb(y)

            h = x.type(self.dtype)
            h = torch.cat([h, ref_net_feats.pop(0)], dim=-1)  # ref net features

            temporal_layer_i = 0
            for module in self.input_blocks:
                h = module(h, emb, context)
                hs.append(h)
                h = h[..., :h.shape[-1] // 2]  # first half
                # === temporal layer ===
                h = rearrange(h, "(b t) c h w -> b c t h w", b=batch_size, t=time_frames)
                h = temporal_layers[temporal_layer_i](h, None, orig_context)  # temporal layer
                temporal_layer_i += 1
                h = rearrange(h, "b c t h w -> (b t) c h w")
                # === temporal layer ===
                h = torch.cat([h, ref_net_feats.pop(0)], dim=-1)  # ref net features
            h = self.middle_block(h, emb, context)
            h = h[..., :h.shape[-1] // 2]  # first half
            # === temporal layer ===
            h = rearrange(h, "(b t) c h w -> b c t h w", b=batch_size, t=time_frames)
            h = temporal_layers[temporal_layer_i](h, None, orig_context)  # temporal layer
            temporal_layer_i += 1
            h = rearrange(h, "b c t h w -> (b t) c h w")
            # === temporal layer ===
            h = torch.cat([h, ref_net_feats.pop(0)], dim=-1)  # ref net features
            for module in self.output_blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = module(h, emb, context)
                h = h[..., :h.shape[-1] // 2]  # first half
                # === temporal layer ===
                h = rearrange(h, "(b t) c h w -> b c t h w", b=batch_size, t=time_frames)
                h = temporal_layers[temporal_layer_i](h, None, orig_context)  # temporal layer
                temporal_layer_i += 1
                h = rearrange(h, "b c t h w -> (b t) c h w")
                # === temporal layer ===
                h = torch.cat([h, ref_net_feats.pop(0)], dim=-1)  # ref net features
            h = h.type(x.dtype)

            h = h[..., :h.shape[-1] // 2]  # first half
            assert len(ref_net_feats) == 0
            assert temporal_layer_i == len(temporal_layers)

            return self.out(h)
        
        self.sd_unet_hack_forward_phase_2 = hack_forward_phase_2

    def forward(self, x, clip_features, reference_net_feats, timesteps,
                batch_size, time_frames, phase):
        # for phase==1
        if phase == 1:
            pred_noise = self.sd_unet_hack_forward(self.sd_unet, x, timesteps, ref_net_feats=reference_net_feats,
                                                context=clip_features["c_crossattn"], y=clip_features["c_adm"])
            return pred_noise
        elif phase == 2:
            # repeat these features for each time frame
            orig_context = clip_features["c_crossattn"]
            context = repeat(clip_features["c_crossattn"], "b x y -> (b t) x y", t=time_frames)
            y = repeat(clip_features["c_adm"], "b x -> (b t) x", t=time_frames)
            for i in range(len(reference_net_feats)):
                reference_net_feats[i] = repeat(reference_net_feats[i], "b c h w -> (b t) c h w", t=time_frames)

            pred_noise = self.sd_unet_hack_forward_phase_2(self.sd_unet, x, timesteps, ref_net_feats=reference_net_feats,
                                                           context=context, y=y,
                                                           temporal_layers=self.temporal_layers, batch_size=batch_size, time_frames=time_frames, orig_context=orig_context)
            return pred_noise
        else:
            raise ValueError("phase should be 1 or 2")
