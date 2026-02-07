import math
import numpy as np
import torch
from torchvision import transforms
from PIL import Image, ImageOps


def resize_and_fg_crop(image, mask, target_width, target_height):
    pil_img = Image.fromarray(image).convert("RGB")

    mask_np = np.squeeze(mask)

    if mask_np.dtype != np.uint8:
        mask_np = (mask_np * 255).astype(np.uint8)

    pil_mask = Image.fromarray(mask_np).convert("L")

    # NOTE: don't invert mask here. 
    # inverted_mask = ImageOps.invert(pil_mask)
    # bbox = inverted_mask.getbbox()

    bbox = pil_mask.getbbox()
    # print("bbox is", bbox)

    # crop BOTH using the SAME bbox
    if bbox:
        pil_img  = pil_img.crop(bbox)
        pil_mask = pil_mask.crop(bbox)

    crop_sizes = pil_img.size

    # # debug
    # pil_img.save("fg_crop.png")
    # pil_mask.save("mask_crop.png")

    # resize BOTH to target
    pil_img  = pil_img.resize((target_width, target_height), Image.LANCZOS)
    pil_mask = pil_mask.resize((target_width, target_height), Image.NEAREST)

    # # debug
    # pil_img.save("fg_crop_resize.png")
    # pil_mask.save("mask_crop_resize.png")

    fg_512 = np.array(pil_img)                      # HxWx3 uint8
    m_512  = np.array(pil_mask).astype(np.uint8)    # HxW uint8 0..255

    return fg_512, m_512, bbox, crop_sizes


@torch.inference_mode()
def encode_prompt_inner(txt: str, tokenizer, text_encoder, device):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids

    token_ids = torch.tensor(tokens).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids)[0]

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt, tokenizer, text_encoder, device):
    c = encode_prompt_inner(positive_prompt, tokenizer, text_encoder, device)
    uc = encode_prompt_inner(negative_prompt, tokenizer, text_encoder, device)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def pytorch2numpy_new(imgs, quant=True):
    if quant:
        results = []
        for img in imgs:
            img = img * 0.5 + 0.5     # Unnormalize
            img = img.movedim(0, -1) * 255
            img = img.clamp(0, 255)     # Clamp values to be between 0 and 1
            img = img.detach().float().cpu().numpy().astype(np.uint8)         # Convert back to numpy array
            results.append(img)
    else:
        results = []
        for img in imgs:
            img = img * 0.5 + 0.5     # Unnormalize
            img = img.movedim(0, -1)
            img = img.clamp(0, 1)     # Clamp values to be between 0 and 1
            img = img.detach().float().cpu().numpy().astype(np.float32)         # Convert back to numpy array
            results.append(img)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


@torch.inference_mode()
def numpy2pytorch_new(imgs):
    # h = torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0  # so that 127 must be strictly 0.0
    # h = h.movedim(-1, 1)
    transform1 = transforms.ToTensor()
    transform2 = transforms.Normalize((.5,.5,.5), (.5,.5,.5))
    hs = []
    for img in imgs:
        h = transform1(img)
        h = transform2(h)
        hs.append(h)
    h = torch.stack(hs, dim=0)
    return h