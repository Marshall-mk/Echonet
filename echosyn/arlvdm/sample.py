import argparse
import logging
import math
import os
import shutil
import json
from glob import glob
from einops import rearrange
from omegaconf import OmegaConf
import numpy as np
from tqdm import tqdm
from packaging import version
from functools import partial
from PIL import Image
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet3DConditionModel, UNetSpatioTemporalConditionModel

from echosyn.common.datasets import TensorSet, ImageSet, TensorSequenceSet
from echosyn.common import (
        pad_reshape, unpad_reshape, padf, unpadf, 
        load_model, save_as_mp4, save_as_gif, save_as_img, save_as_avi,
        parse_formats,
    )

"""
CUDA_VISIBLE_DEVICES='6' python echosyn/arlvdm/samplex.py \
    --config echosyn/arlvdm/configs/default.yaml \
    --unet experiments/arlvdm/checkpoint-500000/unet_ema \
    --vae models/vae \
    --conditioning samples/lidm_dynamic/privacy_compliant_latents\
    --output samples/arlvdm_dynamic64 \
    --num_samples 2048 \
    --batch_size 8 \
    --num_steps 64 \
    --min_lvef 10 \
    --max_lvef 90 \
    --save_as mp4,jpg \
    --frames 192 \
    --prior_frames 64\
    --stride 32 \
    --seed 42
"""

if __name__ == "__main__":
    # 1 - Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config file.")
    parser.add_argument("--unet", type=str, default=None, help="Path unet checkpoint.")
    parser.add_argument("--vae", type=str, default=None, help="Path vae checkpoint.")
    parser.add_argument("--conditioning", type=str, default=None, help="Path to the folder containing the conditionning latents.")
    parser.add_argument("--output", type=str, default='.', help="Output directory.")
    parser.add_argument("--num_samples", type=int, default=8, help="Number of samples to generate.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument("--num_steps", type=int, default=64, help="Number of steps.")
    parser.add_argument("--min_lvef", type=int, default=10, help="Minimum LVEF.")
    parser.add_argument("--max_lvef", type=int, default=90, help="Maximum LVEF.")
    parser.add_argument("--save_as", type=parse_formats, default=None, help="Save formats separated by commas (e.g., avi,jpg). Available: avi, mp4, gif, jpg, png, pt")
    parser.add_argument("--frames", type=int, default=192, help="Number of frames to generate.")
    parser.add_argument("--prior_frames", type=int, default=64, help="Number of prior frames to use for conditioning.")
    parser.add_argument("--stride", type=int, default=32, help="Number of frames to generate in each step. Use 1 for fully autoregressive.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)

    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # 2 - Load models
    unet = load_model(args.unet)
    vae = load_model(args.vae)

    # 3 - Load scheduler
    scheduler_kwargs = OmegaConf.to_container(config.noise_scheduler)
    scheduler_klass_name = scheduler_kwargs.pop("_class_name")
    scheduler_klass = getattr(diffusers, scheduler_klass_name, None)
    assert scheduler_klass is not None, f"Could not find scheduler class {scheduler_klass_name}"
    scheduler = scheduler_klass(**scheduler_kwargs)
    scheduler.set_timesteps(args.num_steps)
    timesteps = scheduler.timesteps

    # 4 - Load dataset
    ## detect type of conditioning:
    file_ext = os.listdir(args.conditioning)[0].split(".")[-1].lower()
    assert file_ext in ["pt", "jpg", "png"], f"Conditioning files must be either .pt, .jpg or .png, not {file_ext}"
    if file_ext == "pt":
        dataset = TensorSet(args.conditioning)
        # dataset = TensorSequenceSet(args.conditioning, seq_len=args.prior_frames)
    else:
        dataset = ImageSet(args.conditioning, ext=file_ext)
    assert len(dataset) > 0, f"No files found in {args.conditioning} with extension {file_ext}"

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, drop_last=True)

    # 5 - Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float32
    generator = torch.Generator(device=device).manual_seed(args.seed) if args.seed is not None else None
    unet = unet.to(device, dtype)
    vae = vae.to(device, torch.float32)
    unet.eval()
    vae.eval()

    format_input = pad_reshape if config.unet._class_name == "UNetSpatioTemporalConditionModel" else padf
    format_output = unpad_reshape if config.unet._class_name == "UNetSpatioTemporalConditionModel" else unpadf

    B, C = args.batch_size, config.unet.out_channels
    H, W = config.unet.sample_size, config.unet.sample_size
    fps = config.globals.target_fps
    prior_frames = args.prior_frames
    total_frames = args.frames
    stride = args.stride

    forward_kwargs = {
        "timestep": -1,
    }

    if config.unet._class_name == "UNetSpatioTemporalConditionModel":
        dummy_added_time_ids = torch.zeros((B, config.unet.addition_time_embed_dim), device=device, dtype=dtype)
        forward_kwargs["added_time_ids"] = dummy_added_time_ids
    
    sample_index = 0
    filelist = []

    os.makedirs(args.output, exist_ok=True)
    for ext in args.save_as:
        os.makedirs(os.path.join(args.output, ext), exist_ok=True)
    finished = False

    pbar = tqdm(total=args.num_samples)

    # 6 - Generate samples
    with torch.no_grad():
        while not finished:
            for cond in dataloader:
                # cond = cond.permute(0, 2, 1, 3, 4) # with TensorSequenceSet
                if finished:
                    break

                # Prepare conditioning - lvef
                lvefs = torch.randint(args.min_lvef, args.max_lvef+1, (B,), device=device, dtype=dtype, generator=generator)
                lvefs = lvefs / 100.0
                lvefs = lvefs[:, None, None]
                forward_kwargs["encoder_hidden_states"] = lvefs

                # Prepare conditioning - reference frames
                latent_cond_images = cond.to(device, torch.float32)
                if file_ext != "pt":
                    # project image to latent space
                    latent_cond_images = vae.encode(latent_cond_images).latent_dist.sample()
                    latent_cond_images = latent_cond_images * vae.config.scaling_factor
                
                # Initialize the frames buffer with the conditioning frames
                all_frames = []
                
                # Initialize conditioning frames - expand to match the expected dimensions
                conditioning_frames = latent_cond_images[:,:,None,:,:].repeat(1, 1, prior_frames, 1, 1) # B x C x T x H x W
                # conditioning_frames = latent_cond_images # B x C x T x H x W # with TensorSequenceSet
                
                
                # Generate frames autoregressively with the specified stride
                for frame_idx in range(0, total_frames, stride):
                    # Generate new frames using the previous frames as conditioning
                    # Prepare latent noise for current_stride frames
                    latents = torch.randn((B, C, prior_frames, H, W), device=device, dtype=dtype, generator=generator)

                    # Denoise the latents to generate new frames
                    with torch.autocast("cuda"):
                        for t in timesteps:
                            forward_kwargs["timestep"] = t
                            latent_model_input = scheduler.scale_model_input(latents, timestep=t)
                            
                            # Concatenate the model input with conditioning frames
                            # For each frame in the batch, we need the appropriate conditioning
                            latent_model_input = torch.cat((latent_model_input, conditioning_frames), dim=1) # B x 2C x current_stride x H x W
                            latent_model_input, padding = format_input(latent_model_input, mult=3)
                            
                            # Model prediction
                            noise_pred = unet(latent_model_input, **forward_kwargs).sample
                            noise_pred = format_output(noise_pred, pad=padding)
                            latents = scheduler.step(noise_pred, t, latents).prev_sample
                    
                    # Store the generated frames
                    all_frames.append(latents[:,:,:stride,:,:]) # B x C x T x H x W
                    # Update conditioning frames with the generated frames
                    conditioning_frames = torch.cat((conditioning_frames[:,:,stride:,:,:], latents[:,:,:stride,:,:]), dim=2)

                
                # Concatenate all generated frames
                video_latents = torch.cat(all_frames, dim=2)  # B x C x T x H x W
                
                # Make sure we have exactly total_frames
                if video_latents.shape[2] > total_frames:
                    video_latents = video_latents[:,:,:total_frames,:,:]
                
                # VAE decode
                latents = rearrange(video_latents, "b c t h w -> (b t) c h w").cpu()
                latents = latents / vae.config.scaling_factor

                # Decode in chunks to save memory
                chunked_latents = torch.split(latents, args.batch_size, dim=0)
                decoded_chunks = []
                for chunk in chunked_latents:
                    decoded_chunks.append(vae.decode(chunk.float().cuda()).sample.cpu())
                video = torch.cat(decoded_chunks, dim=0) # (B*T) x H x W x C

                # format output
                video = rearrange(video, "(b t) c h w -> b t h w c", b=B)
                video = (video + 1) * 128
                video = video.clamp(0, 255).to(torch.uint8)

                logger.info(f"Video shape: {video.shape}, dtype: {video.dtype}, min: {video.min()}, max: {video.max()}")
                file_lvefs = lvefs.squeeze().mul(100).to(torch.int).tolist()
                
                # save samples
                for j in range(B):
                    # FileName,EF,ESV,EDV,FrameHeight,FrameWidth,FPS,NumberOfFrames,Split
                    filelist.append([f"sample_{sample_index:06d}", file_lvefs[j], 0, 0, video.shape[2], video.shape[3], fps, video.shape[1], "TRAIN"])
                    
                    # Wrap each save operation in try-except to handle encoding errors gracefully
                    if "mp4" in args.save_as:
                        try:
                            save_as_mp4(video[j], os.path.join(args.output, "mp4", f"sample_{sample_index:06d}.mp4"))
                        except Exception as e:
                            logger.error(f"Failed to save as MP4: {e}")
                    
                    if "avi" in args.save_as:
                        try:
                            save_as_avi(video[j], os.path.join(args.output, "avi", f"sample_{sample_index:06d}.avi"))
                        except Exception as e:
                            logger.error(f"Failed to save as AVI: {e}")
                    
                    if "gif" in args.save_as:
                        try:
                            save_as_gif(video[j], os.path.join(args.output, "gif", f"sample_{sample_index:06d}.gif"))
                        except Exception as e:
                            logger.error(f"Failed to save as GIF: {e}")
                    
                    if "jpg" in args.save_as:
                        try:
                            save_as_img(video[j], os.path.join(args.output, "jpg", f"sample_{sample_index:06d}"), ext="jpg")
                        except Exception as e:
                            logger.error(f"Failed to save as JPG: {e}")
                    
                    if "png" in args.save_as:
                        try:
                            save_as_img(video[j], os.path.join(args.output, "png", f"sample_{sample_index:06d}"), ext="png")
                        except Exception as e:
                            logger.error(f"Failed to save as PNG: {e}")
                    
                    if "pt" in args.save_as:
                        try:
                            torch.save(video[j].clone(), os.path.join(args.output, "pt", f"sample_{sample_index:06d}.pt"))
                        except Exception as e:
                            logger.error(f"Failed to save as PT: {e}")
                    
                    sample_index += 1
                    pbar.update(1)
                    if sample_index >= args.num_samples:
                        finished = True
                        break

    df = pd.DataFrame(filelist, columns=["FileName", "EF", "ESV", "EDV", "FrameHeight", "FrameWidth", "FPS", "NumberOfFrames", "Split"])
    df.to_csv(os.path.join(args.output, "FileList.csv"), index=False)
    print(f"Generated {sample_index} samples.")
