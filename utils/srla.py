"""Image-domain spectral retention for training-only target assignment."""

import math

import torch
import torch.nn.functional as F


class SRLAAnalyzer:
    """Compare target and surrounding background spectra after image-space stride sampling.

    Unmasked rectangular background strips avoid the artificial internal edges of binary ROI masks.
    Each patch is mean-centered, Hann-windowed and represented by four radial log-amplitude bands.
    These descriptors are a sampling proxy, not a measurement of learned feature information.
    """

    def __init__(self, strides, size=64, context=1.5, min_contrast=1e-4):
        if size < 8 or context <= 1 or min_contrast <= 0:
            raise ValueError("SRLA requires size >= 8, context > 1 and min_contrast > 0")
        self.strides = tuple(float(s) for s in strides)
        if not self.strides or any(s <= 0 for s in self.strides):
            raise ValueError("SRLA strides must be positive")
        self.size, self.context, self.min_contrast = size, context, min_contrast
        window = torch.hann_window(size, periodic=False)
        self.window = window[:, None] * window[None, :]
        radius = torch.sqrt(torch.fft.fftfreq(size)[:, None] ** 2 + torch.fft.rfftfreq(size)[None, :] ** 2)
        bands = (radius / math.sqrt(0.5) * 4).long().clamp(max=3)
        self.bands = torch.stack([bands == k for k in range(4)]).float()
        # rfft omits conjugate frequencies, except the DC and Nyquist columns.
        self.bands[..., 1:] *= 2
        if size % 2 == 0:
            self.bands[..., -1] /= 2

    def descriptor(self, patches):
        """Return batched normalized radial descriptors; constant patches occupy the lowest band."""
        patches = torch.cat(patches)
        window = self.window
        mean = (patches * window).sum((-2, -1), keepdim=True) / window.sum()
        spectrum = torch.fft.rfft2((patches - mean) * window).abs().log1p()
        energy = torch.einsum("nchw,khw->nk", spectrum, self.bands)
        mass = energy.sum(1, keepdim=True)
        descriptors = energy / mass.clamp_min(1e-12)
        flat = (patches.amax((-2, -1)) - patches.amin((-2, -1)))[:, 0] <= 1e-6
        # A flat patch is represented by the lowest band, avoiding numerical FFT noise.
        descriptors[flat] = descriptors.new_tensor([1, 0, 0, 0])
        return descriptors

    @torch.no_grad()
    def __call__(self, images, targets):
        """Analyze final augmented RGB images and normalized (image,class,x,y,w,h) targets.

        Crops remain at their actual pixel size during area downsampling and bilinear reconstruction.
        Only the spectral descriptors use a common square size. Undefined baseline comparisons
        return neutral retention (ones), preserving the original anchor assignments.
        """
        retention = images.new_ones((len(targets), len(self.strides)), dtype=torch.float32)
        if not len(targets):
            return retention
        self.window = self.window.to(images.device)
        self.bands = self.bands.to(images.device)
        gray = (images.float() * images.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
        height, width = images.shape[-2:]
        # One CPU transfer avoids per-coordinate GPU synchronization for variable-size crops.
        boxes = targets.detach().float().cpu().tolist()
        # Bound FFT memory while batching comparisons across objects, contexts and strides.
        for start in range(0, len(boxes), 64):
            patches, object_ids, context_ids, context_owners, valid_ids, context_counts = [], [], [], [], [], []
            for i, (batch, _, x, y, w, h) in enumerate(boxes[start : start + 64], start):
                x, w, y, h = x * width, w * width, y * height, h * height
                x1, x2 = max(0, math.floor(x - w / 2)), min(width, math.ceil(x + w / 2))
                y1, y2 = max(0, math.floor(y - h / 2)), min(height, math.ceil(y + h / 2))
                cx1, cx2 = max(0, math.floor(x - w * self.context / 2)), min(width, math.ceil(x + w * self.context / 2))
                cy1, cy2 = (
                    max(0, math.floor(y - h * self.context / 2)),
                    min(height, math.ceil(y + h * self.context / 2)),
                )
                crop = gray[int(batch) : int(batch) + 1, :, cy1:cy2, cx1:cx2]
                ch, cw = crop.shape[-2:]
                ox1, ox2, oy1, oy2 = x1 - cx1, x2 - cx1, y1 - cy1, y2 - cy1
                if ox2 - ox1 < 2 or oy2 - oy1 < 2:
                    continue
                context = [(0, 0, cw, oy1), (0, oy2, cw, ch), (0, oy1, ox1, oy2), (ox2, oy1, cw, oy2)]
                context = [r for r in context if r[2] - r[0] >= 2 and r[3] - r[1] >= 2]
                if not context:
                    continue
                valid_ids.append(i)
                regions = [(ox1, oy1, ox2, oy2)] + context
                for stride in (None,) + self.strides:
                    if stride is None:
                        restored = crop
                    else:
                        sampled = F.interpolate(
                            crop, size=(max(1, math.ceil(ch / stride)), max(1, math.ceil(cw / stride))), mode="area"
                        )
                        restored = F.interpolate(sampled, size=(ch, cw), mode="bilinear", align_corners=False)
                    owner = len(object_ids)
                    object_ids.append(len(patches))
                    context_ids.extend(range(len(patches) + 1, len(patches) + len(regions)))
                    context_owners.extend([owner] * len(context))
                    context_counts.append(len(context))
                    for rx1, ry1, rx2, ry2 in regions:
                        patches.append(
                            F.interpolate(restored[..., ry1:ry2, rx1:rx2], size=(self.size, self.size), mode="area")
                        )
            if not patches:
                continue
            descriptors = self.descriptor(patches)
            obj = descriptors[object_ids]
            ctx = torch.zeros_like(obj)
            owners = torch.tensor(context_owners, device=images.device)[:, None].expand(-1, 4)
            ctx.scatter_add_(0, owners, descriptors[context_ids])
            ctx /= ctx.new_tensor(context_counts)[:, None]
            midpoint = (obj + ctx) / 2
            distance = 0.5 * (
                (obj * (obj.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(1)
                + (ctx * (ctx.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(1)
            )
            distance = distance.view(-1, len(self.strides) + 1)
            baseline = distance[:, :1]
            scores = (distance[:, 1:] / baseline.clamp_min(self.min_contrast)).clamp(0, 1)
            retention[valid_ids] = torch.where(baseline >= self.min_contrast, scores, torch.ones_like(scores))
        return retention
