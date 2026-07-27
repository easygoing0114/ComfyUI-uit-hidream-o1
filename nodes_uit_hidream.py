"""
nodes_uit_hidream.py
────────────────────────────────────────────────────────────────────────────
UIT Sampler — Sampling node for HiDream-O1-Image (noise_scale float support).
Use the standard Load Checkpoint node to load the model.

Migrated to ComfyUI V3 node schema (comfy_api.latest).

Installation
────────────
Place this file in:
    ComfyUI/custom_nodes/nodes_uit_hidream.py
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

from comfy_api.latest import ComfyExtension, io, ui

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET_MEGAPIXELS = 4_000_000  # 4 MP training resolution
PATCH_MULTIPLE = 32            # resolution must be divisible by this
DEFAULT_WIDTH = 2048
DEFAULT_HEIGHT = 2048


# ─────────────────────────────────────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────────────────────────────────────

def _round_to_multiple(value: int, multiple: int = PATCH_MULTIPLE) -> int:
    """Round down to the nearest multiple of `multiple` (minimum = multiple)."""
    return max(multiple, (value // multiple) * multiple)


def _rescale_image_to_megapixels(
    image_tensor: torch.Tensor,
    target_mp: int = TARGET_MEGAPIXELS,
    multiple: int = PATCH_MULTIPLE,
) -> torch.Tensor:
    """
    Rescale a ComfyUI IMAGE tensor (B, H, W, C) to ~target_mp pixels while
    preserving aspect ratio. Both sides are rounded down to `multiple`.
    Uses Lanczos resampling via PIL.
    """
    B, H, W, C = image_tensor.shape
    aspect = W / H

    new_h = _round_to_multiple(round(math.sqrt(target_mp / aspect)))
    new_w = _round_to_multiple(round(aspect * math.sqrt(target_mp / aspect)))

    if new_h == H and new_w == W:
        return image_tensor

    device = image_tensor.device
    dtype = image_tensor.dtype

    out_frames = []
    for b in range(B):
        frame_np = (image_tensor[b].clamp(0, 1).cpu().float().numpy() * 255).astype("uint8")
        mode = "RGBA" if C == 4 else "RGB"
        pil_img = PILImage.fromarray(frame_np, mode=mode)
        pil_img = pil_img.resize((new_w, new_h), PILImage.LANCZOS)
        resized = torch.from_numpy(np.array(pil_img).astype("float32") / 255.0)
        out_frames.append(resized)

    return torch.stack(out_frames, dim=0).to(device=device, dtype=dtype)


# ─────────────────────────────────────────────────────────────────────────────
# UIT Sampler
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLER_NAMES = comfy.samplers.KSampler.SAMPLERS
_SCHEDULER_NAMES = comfy.samplers.KSampler.SCHEDULERS


class UITSampler(io.ComfyNode):
    """
    UIT Sampler
    ──────────────────────────────────────────────────────────────────────────
    All-in-one sampling node for HiDream-O1-Image.
    Intermediate step images are available in bulk from 'step_images' output.
    """

    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="UITSampler",
            display_name="UIT Sampler",
            category="sampling/uit",
            description=(
                "All-in-one sampling node for HiDream-O1-Image. "
                "Intermediate step images are available in bulk from the "
                "'step_images' output."
            ),
            inputs=[
                io.Model.Input("model"),
                # ── generation settings ───────────────────────────────────
                io.Int.Input(
                    "width",
                    default=DEFAULT_WIDTH,
                    min=64, max=4096, step=32,
                    tooltip="Ignored when input_image is connected.",
                ),
                io.Int.Input(
                    "height",
                    default=DEFAULT_HEIGHT,
                    min=64, max=4096, step=32,
                    tooltip="Ignored when input_image is connected.",
                ),
                io.Int.Input(
                    "seed",
                    default=0,
                    min=0, max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                ),
                io.Float.Input(
                    "cfg",
                    default=3.0,
                    min=0.0, max=30.0, step=0.1,
                ),
                io.Combo.Input(
                    "sampler",
                    options=_SAMPLER_NAMES,
                    default="euler",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=_SCHEDULER_NAMES,
                    default="normal",
                ),
                io.Int.Input(
                    "steps",
                    default=12,
                    min=1, max=200,
                ),
                io.Float.Input(
                    "denoise",
                    default=1.0,
                    min=0.0, max=1.0, step=0.01,
                ),
                io.Float.Input(
                    "noise_scale",
                    default=8.0,
                    min=1.0, max=12.0, step=0.1,
                    tooltip="Equivalent to ModelNoiseScale. HiDream-O1 base: 8.0, dev: 7.5.",
                ),
                # ── optional inputs ────────────────────────────────────────
                io.Clip.Input("clip", optional=True),
                io.Vae.Input("vae", optional=True),
                io.Image.Input(
                    "input_image",
                    optional=True,
                    tooltip="Source image for img2img. Rescaled to 4MP via Lanczos.",
                ),
                io.Image.Input(
                    "reference_image1",
                    optional=True,
                    tooltip="Reference image 1.",
                ),
                io.Image.Input(
                    "reference_image2",
                    optional=True,
                    tooltip="Reference image 2.",
                ),
                io.String.Input(
                    "positive_prompt",
                    optional=True,
                    force_input=True,
                ),
                io.String.Input(
                    "negative_prompt",
                    optional=True,
                    force_input=True,
                ),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
                io.Image.Output(display_name="step_images"),
            ],
        )

    # ── Internal helpers ──────────────────────────────────────────────────────
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

    # ── Main execution ────────────────────────────────────────────────────────
    @classmethod
    def execute(
        cls,
        model,
        width: int,
        height: int,
        seed: int,
        cfg: float,
        sampler: str,
        scheduler: str,
        steps: int,
        denoise: float,
        noise_scale: float,
        clip=None,
        vae=None,
        input_image: Optional[torch.Tensor] = None,
        reference_image1: Optional[torch.Tensor] = None,
        reference_image2: Optional[torch.Tensor] = None,
        positive_prompt: str = "",
        negative_prompt: str = "",
    ) -> io.NodeOutput:
        # ── 0. UiT model detection log ────────────────────────────────────────
        try:
            model_class_name = type(model.model).__name__
            if "HiDreamO1" in model_class_name or "UiT" in model_class_name:
                print(
                    f"[UITSampler] UiT model detected ({model_class_name}). "
                    "Loading dummy CLIP and VAE."
                )
        except Exception:
            pass

        # ── 1. Apply noise scale ──────────────────────────────────────────────
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

        # ── 2. Encode prompts ─────────────────────────────────────────────────
        positive_cond = cls._encode_prompt(clip, positive_prompt)
        negative_cond = cls._encode_prompt(clip, negative_prompt)

        # ── 3. Attach reference images to conditioning ────────────────────────
        refs = [r for r in (reference_image1, reference_image2) if r is not None]
        if refs:
            positive_cond = node_helpers.conditioning_set_values(
                positive_cond, {"reference_latents": refs}, append=True
            )
            negative_cond = node_helpers.conditioning_set_values(
                negative_cond, {"reference_latents": refs}, append=True
            )

        # ── 4. Prepare latent ─────────────────────────────────────────────────
        if input_image is not None:
            rescaled = _rescale_image_to_megapixels(input_image)
            latent_dict = cls._image_to_latent(vae, rescaled)
        else:
            gen_w = _round_to_multiple(width)
            gen_h = _round_to_multiple(height)
            latent_dict = cls._latent_from_size(gen_w, gen_h)

        # ── 5. Build sampler and sigmas ───────────────────────────────────────
        sampler_obj = comfy.samplers.sampler_object(sampler)
        model_sampling = m.get_model_object("model_sampling")

        if denoise <= 0.0:
            sigmas = torch.FloatTensor([])
        else:
            total_steps = steps if denoise >= 1.0 else int(steps / denoise)
            sigmas = comfy.samplers.calculate_sigmas(
                model_sampling, scheduler, total_steps
            ).cpu()
            sigmas = sigmas[-(steps + 1):]

        # ── 6. Run sampling (with step image collection callback) ─────────────
        noise = comfy.sample.prepare_noise(latent_dict["samples"], seed, None)

        x0_output = {}
        preview_steps = max(len(sigmas) - 1, 1)
        base_callback = latent_preview.prepare_callback(m, preview_steps, x0_output)

        step_images_list = []

        def uitsampler_callback(step, x0, x, total_steps):
            if base_callback is not None:
                base_callback(step, x0, x, total_steps)
            if x0 is not None:
                with torch.no_grad():
                    decoded = vae.decode(x0)
                    step_images_list.append(decoded.cpu())

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
            callback=uitsampler_callback,
            disable_pbar=False,
            seed=seed,
        )

        # ── 7. Decode → IMAGE tensor ──────────────────────────────────────────
        image_out = vae.decode(samples_out)

        # ── 8. Pack step_images ───────────────────────────────────────────────
        if step_images_list:
            step_images = torch.cat(step_images_list, dim=0).to(device=image_out.device)
        else:
            step_images = image_out.clone()

        return io.NodeOutput(image_out, step_images)


# ─────────────────────────────────────────────────────────────────────────────
# ComfyUI registration (V3)
# ─────────────────────────────────────────────────────────────────────────────

class UITHiDreamExtension(ComfyExtension):
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            UITSampler,
        ]


async def comfy_entrypoint() -> UITHiDreamExtension:
    return UITHiDreamExtension()