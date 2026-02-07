import os
import math
import numpy as np
import torch
import safetensors.torch as sf
# import db_examples
from torchvision import transforms
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler, EulerAncestralDiscreteScheduler, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPTextModel, CLIPTokenizer
from briarmbg import BriaRMBG
from enum import Enum
from torch.hub import download_url_to_file
import cv2
from ip_adapter import IPAdapterPlus
from tqdm import tqdm
import sys
sys.path.append("SD15/FixDetails")

from PIL import Image, ImageOps

from util import (
    encode_prompt_pair,
    pytorch2numpy, 
    pytorch2numpy_new,
    numpy2pytorch, 
    numpy2pytorch_new,
    resize_and_fg_crop,
)

from detail_fixer import DetailsFixer
# detail_fixer = DetailsFixer(model_path="xxx/vqmodel")
detail_fixer = DetailsFixer(model_path="ckpt/SD15/vqmodel")


# adapter
image_encoder_path = "ckpt/CLIP/models"
# ip_ckpt = 'ckpt/SD15/ip_ckpt/adapter.bin'
ip_ckpt = 'ckpt/SD15/adapter.bin'

device = "cuda"

INVERT_MASK = True # TODO: invert the mask to be white fg and black bg

sd15_name = 'stablediffusionapi/realistic-vision-v51'
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(sd15_name, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet", revision=None)
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")


# Change UNet
with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(12, unet.conv_in.out_channels, unet.conv_in.kernel_size, unet.conv_in.stride, unet.conv_in.padding)
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in
unet_original_forward = unet.forward

def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs['cross_attention_kwargs']['concat_conds'].to(sample)
    c_concat = torch.cat([c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0)
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs['cross_attention_kwargs'] = {}
    return unet_original_forward(new_sample, timestep, encoder_hidden_states, **kwargs)


unet.forward = hooked_unet_forward

# Load
model_path = "ckpt/SD15/model.safetensors"

# if not os.path.exists(model_path):
#     download_url_to_file(url='https://huggingface.co/lllyasviel/ic-light/resolve/main/iclight_sd15_fbc.safetensors', dst=model_path)

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] * 0 + sd_offset['unet.' + k] for k in sd_origin.keys()}
# sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device('cuda')
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=ddim_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None
)

ip_model = IPAdapterPlus(t2i_pipe, image_encoder_path, ip_ckpt, device, num_tokens=16)



@torch.inference_mode()
def process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, matting, use_fix=True):
    # bg_source: text or image

    rng = torch.Generator(device=device).manual_seed(seed)
    image_height, image_width = 512, 512
    fg, matting_u8, bbox, crop_size = resize_and_fg_crop(input_fg, matting, image_width, image_height)
    bg = np.zeros_like(fg)

    # restore old RGB mask format
    matting = np.repeat(matting_u8[:, :, None], 3, axis=2)

    matting_array = np.zeros(matting.shape[:2])
    matting_array[matting[:, :, 0] > 128] = 1
    matting_array = np.uint8(matting_array * 255)

    # mask
    matting = numpy2pytorch_new([matting/255]).to(device=vae.device, dtype=vae.dtype)
    matting = matting * 0.5 + 0.5   # (b, c, h, w)
    matting = torch.mean(matting, dim=1, keepdim=True)    # (b, 1, H, W)
    # matting = (matting > 0.5).float()
    # print(matting.sum())

    fg_array = fg.copy()
    
    
    concat_conds = numpy2pytorch_new([fg, bg]).to(device=vae.device, dtype=vae.dtype)
    
    concat_conds[0] = matting * concat_conds[0]
    concat_conds[1] = (1 - matting) * concat_conds[1]
    if bg_source == 'text':
        concat_conds[1] = torch.zeros_like(matting) * concat_conds[1]
        bg = np.ones_like(bg) * 127

    # adapter
    pil_bg = Image.fromarray(bg).convert("RGB")


    fg, bg = pytorch2numpy_new(concat_conds)
    concat_conds = vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    # concat_conds = vae.encode(concat_conds).latent_dist.mode()
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    conds, unconds = encode_prompt_pair(
                                        positive_prompt=prompt + ', ' + a_prompt, 
                                        negative_prompt=n_prompt,
                                        tokenizer=tokenizer,
                                        text_encoder=text_encoder,
                                        device=device,
                                        )

    
    latents = ip_model.generate_new(
        pil_image=pil_bg, 
        prompt_embeds=conds,
        negative_prompt_embeds=unconds,
        width=image_width,
        height=image_height,
        num_inference_steps=steps,
        num_images_per_prompt=num_samples,
        generator=rng,
        output_type='latent',
        guidance_scale=cfg,
        cross_attention_kwargs={'concat_conds': concat_conds},
        matting=matting,
        ).to(vae.dtype) / vae.config.scaling_factor

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy_new(pixels, quant=False)
    for i, p in enumerate(pixels):
        if use_fix:
            fixed_p = detail_fixer.run_mixres(fg_array, np.uint8(p*255), matting_array)
            pixels[i] = fixed_p
        else:
            pixels[i] = np.uint8(p*255)
        # pixels[i] = np.uint8(p*255)
    # return pixels, [fg, bg]
    return pixels, [fg, bg], crop_size

@torch.inference_mode()
def process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, gt_mask_path):
    # input_fg, matting = run_rmbg(input_fg)
    # print(matting.shape, np.unique(matting))
    # exit()

    # use gt mask
    gt_mask = cv2.imread(gt_mask_path, 0)

    # NOTE: invert mask if demanded
    if INVERT_MASK: gt_mask = 255 - gt_mask

    # TODO
    # gt_mask[gt_mask >= 128] = 255
    # gt_mask[gt_mask < 128] = 0
    gt_mask = np.expand_dims(gt_mask, axis=-1) / 255
    # gt_mask[gt_mask > 0] = 1
    # gt_mask[gt_mask < 1] = 0
    matting = gt_mask
    # input_fg = (input_fg * matting).astype(np.uint8)

    results, extra_images, crop_sizes = process(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, matting)
    # results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
    # return results + extra_images
    return results + extra_images, crop_sizes

def inference(input_fg_path, input_bg_path=None, prompt="", image_width=512, image_height=512, num_samples=1, seed=42, steps=25, a_prompt="best quality", n_prompt="low resolution, bad anatomy, bad hands, cropped, worst quality", cfg=3.5, highres_scale=1.5, highres_denoise=0.5, save_root=None, gt_mask_path=None):
    input_fg = cv2.imread(input_fg_path)
    input_fg = cv2.cvtColor(input_fg, cv2.COLOR_BGR2RGB)
    if input_bg_path:
        input_bg = cv2.imread(input_bg_path)
        input_bg = cv2.cvtColor(input_bg, cv2.COLOR_BGR2RGB)
        bg_source="image"
    else:
        input_bg = None
        bg_source="text"
    # all_outputs[0]: relit, all_outputs[1]: fg, all_outputs[2]: bg
    all_outputs, crop_sizes = process_relight(input_fg, input_bg, prompt, image_width, image_height, num_samples, seed, steps, a_prompt, n_prompt, cfg, highres_scale, highres_denoise, bg_source, gt_mask_path=gt_mask_path)
    relit_result = all_outputs[0]
    relit_result = cv2.cvtColor(relit_result, cv2.COLOR_RGB2BGR)
    fg = all_outputs[1]
    fg = cv2.cvtColor(fg, cv2.COLOR_RGB2BGR)
    bg = all_outputs[2]
    bg = cv2.cvtColor(bg, cv2.COLOR_RGB2BGR)
    if save_root:
        if input_bg_path:
            save_path = os.path.join(save_root, os.path.basename(input_bg_path).split('.png')[0], os.path.basename(input_fg_path))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        else:
            save_path = os.path.join(save_root, os.path.basename(input_fg_path))
        # cv2.imwrite(save_path, relit_result)
        # cv2.imwrite(save_path.replace('.png', '_fg.png'), fg)
        # cv2.imwrite(save_path.replace('.png', '_bg.png'), bg)

        # TODO:
        if gt_mask_path is not None:
            # use gt mask
            gt_mask = cv2.imread(gt_mask_path, 0)
            gt_mask = np.expand_dims(gt_mask, axis=-1) / 255
            # TODO
            # matting = resize_and_center_crop_mask(gt_mask, 512, 512)
            matting_array = np.zeros(matting.shape[:2])
            matting_array[matting[:, :, 0] > 128] = 1
            matting_array = np.uint8(matting_array * 255)

    return relit_result, crop_sizes


def main():
    input_fg_path = "/home/shenzhen/Datasets/VITON/test/image/00017_00.jpg"          # required
    gt_mask_path  = "/home/shenzhen/Datasets/VITON/test/fg_masks/00017_00.png"        # required in your current code
    prompt = "Relit with warm golden sunlight during the late afternoon, casting gentle directional shadows and surrounding the subject in soft amber tones to create a calm, radiant mood."

    out, crop_sizes = inference(
        input_fg_path=input_fg_path,
        input_bg_path=None,     # text-based (no bg)
        prompt=prompt,
        gt_mask_path=gt_mask_path,
    )

    # cv2.imwrite("relit.png", out)
    cv2.imwrite("relit.png", cv2.resize(out, (crop_sizes[0], crop_sizes[1])))


# NOTE: simple version: inference with ONE image. 
if __name__ == "__main__":
    main()
