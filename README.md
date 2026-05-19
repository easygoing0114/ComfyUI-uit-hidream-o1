# ComfyUI-uit-hidream-o1

An all-in-one sampling node (**UIT Sampler**) for [HiDream-O1-Image](https://github.com/HiDream-ai/HiDream-I1) .

## ✨ Features

### 🎛️ UIT Sampler

- Single node covering the full generation pipeline for HiDream-O1-Image
- img2img support — input images are automatically rescaled to 4 MP via Lanczos
- Up to 2 reference images attachable to conditioning
- Detects HiDreamO1 models at runtime and logs a notice about dummy CLIP/VAE
- `noise_scale` parameter with float precision (base: `8.0`, dev: `7.5`)
- `step_images` output — all intermediate denoising steps in one batch tensor

## 🔥 Installation

1. Clone this repository into your ComfyUI `custom_nodes` folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/easygoing0114/ComfyUI-uit-hidream-o1.git
```

2. Restart ComfyUI. The **UIT Sampler** node should now appear in the node search under `sampling/uit`.

## 🔍 Verification

When a HiDream-O1 model is loaded, you should see this message in the console:

```
[UITSampler] UiT model detected (HiDreamO1). Loading dummy CLIP and VAE.
```

## 📋 Node Reference

### UIT Sampler

**Category:** `sampling/uit`

### Inputs

| Name | Type | Required | Description |
|---|---|---|---|
| `model` | MODEL | ✅ | HiDream-O1 model loaded via Load Checkpoint |
| `clip` | CLIP | ✅ | CLIP / text encoder (stub connection for HiDream-O1) |
| `vae` | VAE | ✅ | VAE (stub connection for HiDream-O1) |
| `input_image` | IMAGE | optional | Source image for img2img; auto-rescaled to 4 MP |
| `reference_image1` | IMAGE | optional | Reference image 1 |
| `reference_image2` | IMAGE | optional | Reference image 2 |
| `positive_prompt` | STRING | optional | Positive text prompt |
| `negative_prompt` | STRING | optional | Negative text prompt |

### Settings

| Name | Type | Description |
|---|---|---|
| `width` | INT | Output width in pixels (ignored when `input_image` is connected) |
| `height` | INT | Output height in pixels (ignored when `input_image` is connected) |
| `seed` | INT | Random seed |
| `cfg` | FLOAT | Classifier-free guidance scale |
| `sampler` | ENUM | Sampler name (e.g. `euler`) |
| `scheduler` | ENUM | Scheduler name (e.g. `normal`) |
| `steps` | INT | Number of sampling steps |
| `denoise` | FLOAT | Denoising strength (1.0 = full generation) |
| `noise_scale` | FLOAT | Noise scale — base: `8.0`, dev: `7.5` |

### Outputs

| Name | Type | Description |
|---|---|---|
| `image` | IMAGE | Final generated image |
| `step_images` | IMAGE | All intermediate step images stacked into one batch |

### Notes

**HiDream-O1 does not use an external VAE or CLIP.** The node accepts these as optional stub inputs so it fits naturally into standard ComfyUI workflows.

**Default resolution** is 2048×2048 (4 MP), matching the HiDream-O1 training resolution. When `input_image` is connected, the resolution is derived from the rescaled image.

---

## ⚖️ License

This project is licensed under the [MIT License](LICENSE).

---

## Update History

### 2025.5.19

- Initial release: UIT Sampler for HiDream-O1-Image
