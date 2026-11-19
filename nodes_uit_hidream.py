"""
nodes_uit_hidream.py

UIT Sampler: sampling node for HiDream-O1-Image with a float noise_scale.
Load the model with the standard Load Checkpoint node.

Installation:
    ComfyUI/custom_nodes/nodes_uit_hidream.py

Notes:
* HiDream-O1 runs in bf16 or fp32 only. FP16 checkpoints are cast to bf16 on load.
* Step images are kept on the CPU during sampling and decoded afterwards,
  so the VAE is never loaded while the diffusion model is running.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch

import comfy.model_management
import comfy.pixel_space_convert
import comfy.samplers
import comfy.sample
import comfy.utils
import latent_preview
import node_helpers

from comfy_api.latest import ComfyExtension, io

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

TARGET_MEGAPIXELS = 4_000_000  # training resolution (4 MP)
PATCH_MULTIPLE = 32            # width/height must be divisible by this
DEFAULT_WIDTH = 2048
DEFAULT_HEIGHT = 2048


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _round_to_multiple(value: int, multiple: int = PATCH_MULTIPLE) -> int:
    """Round down to a multiple (minimum: one multiple)."""
    return max(multiple, (value // multiple) * multiple)


def _rescale_image_to_megapixels(
    image_tensor: torch.Tensor,
    target_mp: int = TARGET_MEGAPIXELS,
    multiple: int = PATCH_MULTIPLE,
) -> torch.Tensor:
    """Rescale an IMAGE tensor (B, H, W, C) to ~target_mp pixels, keeping aspect ratio."""
    _, H, W, _ = image_tensor.shape
    aspect = W / H

    new_h = _round_to_multiple(round(math.sqrt(target_mp / aspect)), multiple)
    new_w = _round_to_multiple(round(aspect * math.sqrt(target_mp / aspect)), multiple)

    if new_h == H and new_w == W:
        return image_tensor

    samples = image_tensor.clamp(0, 1).movedim(-1, 1)  # B, C, H, W
    samples = comfy.utils.common_upscale(samples, new_w, new_h, "lanczos", "disabled")
    return samples.movedim(1, -1).clamp(0, 1).to(dtype=image_tensor.dtype)


def _is_pixel_space_vae(vae) -> bool:
    return isinstance(
        getattr(vae, "first_stage_model", None),
        comfy.pixel_space_convert.PixelspaceConversionVAE,
    )


def _decode_to_image(vae, samples: torch.Tensor) -> torch.Tensor:
    """
    Samples (B, C, H, W) -> IMAGE tensor (B, H, W, C) in [0, 1].

    HiDream-O1 is pixel-space, so the "VAE" is an identity mapping. The
    (x + 1) / 2 conversion is done directly in fp32 to avoid loading the VAE.
    """
    if _is_pixel_space_vae(vae):
        if samples.is_nested:
            samples = samples.tensors[0]
        out = (samples.detach().to(dtype=torch.float32) + 1.0) / 2.0
        return out.clamp_(0.0, 1.0).movedim(1, -1).contiguous()
    return vae.decode(samples)


# ─────────────────────────────────────────────────────────────────────────────
# UIT Sampler
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLER_NAMES = comfy.samplers.KSampler.SAMPLERS
_SCHEDULER_NAMES = comfy.samplers.KSampler.SCHEDULERS


class UITSampler(io.ComfyNode):
    """All-in-one sampling node for HiDream-O1-Image."""

    @classmethod
    def define_schema(cls) -> io.Schema:
        # Keep the widget order (width ... noise_scale) unchanged,
        # otherwise saved workflows load values into the wrong widgets.
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
                    default=1.0,
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
                    default=7.6,
                    min=1.0, max=12.0, step=0.1,
                    tooltip="Equivalent to ModelNoiseScale. HiDream-O1 base: 8.0, dev: 7.5.",
                ),
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
        if clip is None:
            raise ValueError("[UITSampler] 'clip' input is required (connect CLIP from Load Checkpoint).")
        if vae is None:
            raise ValueError("[UITSampler] 'vae' input is required (connect VAE from Load Checkpoint).")

        # Log model info
        try:
            log.info(
                "[UITSampler] model=%s compute dtype=%s",
                type(model.model).__name__, model.model.get_dtype_inference(),
            )
        except Exception:
            pass

        # 1. Apply noise scale
        m = model.clone()
        original_ms = m.get_model_object("model_sampling")
        # Compare with the model's current value (HiDream-O1 defaults to 8.0, not 1.0).
        current_ns = float(getattr(original_ms, "noise_scale", 1.0))
        if abs(noise_scale - current_ns) > 1e-6:
            new_ms = type(original_ms)(m.model.model_config)
            new_ms.set_parameters(
                shift=original_ms.shift,
                multiplier=original_ms.multiplier,
            )
            new_ms.set_noise_scale(noise_scale)
            m.add_object_patch("model_sampling", new_ms)

        # 2. Encode prompts
        positive_cond = cls._encode_prompt(clip, positive_prompt)
        negative_cond = cls._encode_prompt(clip, negative_prompt)

        # 3. Attach reference images to conditioning
        refs = [r for r in (reference_image1, reference_image2) if r is not None]
        if refs:
            positive_cond = node_helpers.conditioning_set_values(
                positive_cond, {"reference_latents": refs}, append=True
            )
            negative_cond = node_helpers.conditioning_set_values(
                negative_cond, {"prompt_type": "negative"}
            )
            negative_cond = node_helpers.conditioning_set_values(
                negative_cond, {"reference_latents": refs}, append=True
            )

        # 4. Prepare latent
        if input_image is not None:
            rescaled = _rescale_image_to_megapixels(input_image)
            latent_dict = cls._image_to_latent(vae, rescaled)
        else:
            gen_w = _round_to_multiple(width)
            gen_h = _round_to_multiple(height)
            latent_dict = cls._latent_from_size(gen_w, gen_h)

        # 5. Build sampler and sigmas
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

        # 6. Sample, collecting each step's x0 on the CPU
        noise = comfy.sample.prepare_noise(latent_dict["samples"], seed, None)

        x0_output = {}
        preview_steps = max(len(sigmas) - 1, 1)
        base_callback = latent_preview.prepare_callback(m, preview_steps, x0_output)

        step_x0_list = []

        def uitsampler_callback(step, x0, x, total_steps):
            if base_callback is not None:
                base_callback(step, x0, x, total_steps)
            if x0 is None:
                return
            with torch.no_grad():
                x0_t = x0.tensors[0] if getattr(x0, "is_nested", False) else x0
                step_x0_list.append(x0_t.detach().to("cpu", copy=True))

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

        # 7. Decode final image
        image_out = _decode_to_image(vae, samples_out)

        # 8. Decode step images (after sampling)
        step_images = None
        if step_x0_list:
            decoded = [_decode_to_image(vae, x0_cpu).cpu() for x0_cpu in step_x0_list]
            step_images = torch.cat(decoded, dim=0)
            step_x0_list.clear()
            del decoded
        if step_images is None:
            step_images = image_out.clone()
        step_images = step_images.to(device=image_out.device)

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