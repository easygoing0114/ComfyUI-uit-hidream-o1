"""
UIT Sampler for HiDream-O1-Image (noise_scale float support).
Load the model with the standard Load Checkpoint node.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from PIL import Image as PILImage

import comfy.model_management
import comfy.sd
import comfy.samplers
import comfy.sample
import folder_paths
import latent_preview
import node_helpers

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_MEGAPIXELS = 2048 * 2048   # 4,194,304 — native HiDream-O1 resolution
PATCH_MULTIPLE    = 32
DEFAULT_WIDTH     = 2048
DEFAULT_HEIGHT    = 2048


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _round_to_multiple(value: int, multiple: int = PATCH_MULTIPLE) -> int:
    """Round to the nearest multiple of `multiple` (minimum = multiple)."""
    return max(multiple, round(value / multiple) * multiple)


def _rescale_image_to_megapixels(
    image_tensor: torch.Tensor,
    target_mp: int = TARGET_MEGAPIXELS,
    multiple: int = PATCH_MULTIPLE,
) -> torch.Tensor:
    """
    Rescale a ComfyUI IMAGE tensor (B, H, W, C) to ~target_mp pixels while
    preserving aspect ratio. Both dimensions are rounded to `multiple`.
    Uses Lanczos resampling via PIL.
    """
    B, H, W, C = image_tensor.shape
    aspect = W / H
    new_h = _round_to_multiple(round(math.sqrt(target_mp / aspect)))
    new_w = _round_to_multiple(round(aspect * math.sqrt(target_mp / aspect)))

    if new_h == H and new_w == W:
        return image_tensor

    device = image_tensor.device
    dtype  = image_tensor.dtype
    out_frames = []
    for b in range(B):
        frame_np = (image_tensor[b].clamp(0, 1).cpu().float().numpy() * 255).astype("uint8")
        mode     = "RGBA" if C == 4 else "RGB"
        pil_img  = PILImage.fromarray(frame_np, mode=mode)
        pil_img  = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
        resized  = torch.from_numpy(np.array(pil_img).astype("float32") / 255.0)
        out_frames.append(resized)

    return torch.stack(out_frames, dim=0).to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# UIT Sampler
# ---------------------------------------------------------------------------

_SAMPLER_NAMES   = comfy.samplers.KSampler.SAMPLERS
_SCHEDULER_NAMES = comfy.samplers.KSampler.SCHEDULERS


class UITSampler:
    """
    All-in-one sampling node for HiDream-O1-Image.
    Intermediate decoded images are available from the 'step_images' output.
    """

    CATEGORY     = "sampling/uit"
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("image", "step_images")
    FUNCTION     = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":     ("MODEL",),
                "width":     ("INT",   {"default": DEFAULT_WIDTH,
                                        "min": 64, "max": 4096, "step": 32,
                                        "tooltip": "Ignored when input_image is connected."}),
                "height":    ("INT",   {"default": DEFAULT_HEIGHT,
                                        "min": 64, "max": 4096, "step": 32,
                                        "tooltip": "Ignored when input_image is connected."}),
                "seed":      ("INT",   {"default": 0,
                                        "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "cfg":       ("FLOAT", {"default": 3.0,
                                        "min": 0.0, "max": 30.0, "step": 0.1}),
                "sampler":   (_SAMPLER_NAMES,   {"default": "euler"}),
                "scheduler": (_SCHEDULER_NAMES, {"default": "normal"}),
                "steps":     ("INT",   {"default": 12, "min": 1, "max": 200}),
                "denoise":   ("FLOAT", {"default": 1.0,
                                        "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_scale": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 12.0, "step": 0.1,
                                          "tooltip": "Equivalent to ModelNoiseScale. base: 8.0, dev: 7.5."}),
            },
            "optional": {
                "clip":             ("CLIP",),
                "vae":              ("VAE",),
                "input_image":      ("IMAGE",  {"tooltip": "Source image for img2img. Rescaled to 4MP via Lanczos."}),
                "reference_image1": ("IMAGE",  {"tooltip": "Reference image 1."}),
                "reference_image2": ("IMAGE",  {"tooltip": "Reference image 2."}),
                "positive_prompt":  ("STRING", {"forceInput": True}),
                "negative_prompt":  ("STRING", {"forceInput": True}),
            },
        }

    @staticmethod
    def _encode_prompt(clip, text: str):
        tokens = clip.tokenize(text or "")
        result = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
        cond = result.pop("cond")
        return [[cond, result]]

    @staticmethod
    def _image_to_latent(vae, image_tensor: torch.Tensor) -> dict:
        latent = vae.encode(image_tensor[:, :, :, :3])
        return {"samples": latent}

    @staticmethod
    def _latent_from_size(width: int, height: int, batch: int = 1) -> dict:
        dev = comfy.model_management.intermediate_device()
        latent = torch.zeros((batch, 3, height, width), device=dev)
        return {"samples": latent}

    def sample(
        self,
        model,
        clip,
        vae,
        width: int,
        height: int,
        seed: int,
        cfg: float,
        sampler: str,
        scheduler: str,
        steps: int,
        denoise: float,
        noise_scale: float,
        input_image:      Optional[torch.Tensor] = None,
        reference_image1: Optional[torch.Tensor] = None,
        reference_image2: Optional[torch.Tensor] = None,
        positive_prompt:  str = "",
        negative_prompt:  str = "",
    ):
        # 0. Log UiT model detection
        try:
            model_class_name = type(model.model).__name__
            if "HiDreamO1" in model_class_name or "UiT" in model_class_name:
                print(f"[UITSampler] UiT model detected ({model_class_name}).")
        except Exception:
            pass

        # 1. Apply noise scale
        m = model.clone()
        if noise_scale != 1.0:
            original_ms = m.get_model_object("model_sampling")
            new_ms = type(original_ms)(m.model.model_config)
            new_ms.set_parameters(
                shift=original_ms.shift,
                multiplier=original_ms.multiplier,
            )
            new_ms.set_noise_scale(noise_scale)
            m.add_object_patch("model_sampling", new_ms)

        # 2. Encode prompts
        positive_cond = self._encode_prompt(clip, positive_prompt)
        negative_cond = self._encode_prompt(clip, negative_prompt)

        # 3. Attach reference images to conditioning
        refs = [r for r in (reference_image1, reference_image2) if r is not None]
        if refs:
            positive_cond = node_helpers.conditioning_set_values(
                positive_cond, {"reference_latents": refs}, append=True
            )
            negative_cond = node_helpers.conditioning_set_values(
                negative_cond, {"reference_latents": refs}, append=True
            )

        # 4. Prepare latent
        if input_image is not None:
            rescaled    = _rescale_image_to_megapixels(input_image)
            latent_dict = self._image_to_latent(vae, rescaled)
        else:
            gen_w = _round_to_multiple(width)
            gen_h = _round_to_multiple(height)
            latent_dict = self._latent_from_size(gen_w, gen_h)

        # 5. Build sampler and sigmas
        sampler_obj    = comfy.samplers.sampler_object(sampler)
        model_sampling = m.get_model_object("model_sampling")

        if denoise <= 0.0:
            sigmas = torch.FloatTensor([])
        else:
            total_steps = steps if denoise >= 1.0 else int(steps / denoise)
            sigmas = comfy.samplers.calculate_sigmas(
                model_sampling, scheduler, total_steps
            ).cpu()
            sigmas = sigmas[-(steps + 1):]

        # 6. Run sampling with per-step decode callback
        noise = comfy.sample.prepare_noise(latent_dict["samples"], seed, None)

        x0_output     = {}
        preview_steps = max(len(sigmas) - 1, 1)
        base_callback = latent_preview.prepare_callback(m, preview_steps, x0_output)

        step_images_list: list[torch.Tensor] = []

        def _callback(step, x0, x, total_steps):
            if base_callback is not None:
                base_callback(step, x0, x, total_steps)
            if x0 is not None:
                with torch.no_grad():
                    step_images_list.append(vae.decode(x0).cpu())

        samples_out = comfy.sample.sample_custom(
            m,
            noise,
            cfg,
            sampler_obj,
            sigmas,
            positive_cond,
            negative_cond,
            latent_dict["samples"],
            noise_mask=None,
            callback=_callback,
            disable_pbar=False,
            seed=seed,
        )

        # 7. Decode final latent
        image_out = vae.decode(samples_out)

        # 8. Collect step images
        if step_images_list:
            step_images = torch.cat(step_images_list, dim=0)
        else:
            step_images = image_out.clone()

        return (image_out, step_images)


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "UITSampler": UITSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "UITSampler": "UIT Sampler",
}
