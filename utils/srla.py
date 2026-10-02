"""Image-domain spectral retention for training-only target assignment."""

import math

import torch
import torch.nn.functional as F
from torchvision.extension import _assert_has_ops


class SRLAAnalyzer:
    """Batch native-pixel target/context spectral comparisons by padded ROI size.

    Padding buckets organize work without quantizing actual crop or sampled sizes.
    Background strips, Hann windows and radial log-amplitude/JS definitions are unchanged.
    """

    def __init__(self, strides, size=64, context=1.5, min_contrast=1e-4):
        if size < 8 or context <= 1 or min_contrast <= 0:
            raise ValueError("SRLA requires size >= 8, context > 1 and min_contrast > 0")
        self.strides = tuple(float(s) for s in strides)
        if not self.strides or any(s <= 0 for s in self.strides):
            raise ValueError("SRLA strides must be positive")
        _assert_has_ops()
        self.size, self.context, self.min_contrast = size, context, min_contrast
        window = torch.hann_window(size, periodic=False)
        self.window = window[:, None] * window[None, :]
        radius = torch.sqrt(torch.fft.fftfreq(size)[:, None] ** 2 + torch.fft.rfftfreq(size)[None, :] ** 2)
        bands = (radius / math.sqrt(0.5) * 4).long().clamp(max=3)
        self.bands = torch.stack([bands == k for k in range(4)]).float()
        self.bands[..., 1:] *= 2
        if size % 2 == 0:
            self.bands[..., -1] /= 2

    @staticmethod
    def area_pool(images, boxes, size, actual_size=None):
        """Batch adaptive-area pooling on exclusive-end integer rectangles using FP64 prefix sums.

        Optional actual_size specifies logical output sizes before edge padding to common storage.
        """
        n, k = boxes.shape[:2]
        height, width = size
        if actual_size is None:
            actual_size = boxes.new_tensor(size).expand(n, 2)
        oh, ow = actual_size[:, 0, None, None], actual_size[:, 1, None, None]
        iy = torch.arange(height, device=images.device)[None, None].minimum(oh - 1)
        ix = torch.arange(width, device=images.device)[None, None].minimum(ow - 1)
        bh, bw = (boxes[..., 3] - boxes[..., 1])[..., None], (boxes[..., 2] - boxes[..., 0])[..., None]
        y0 = boxes[..., 1, None] + torch.div(iy * bh, oh, rounding_mode="floor")
        y1 = boxes[..., 1, None] + torch.div((iy + 1) * bh + oh - 1, oh, rounding_mode="floor")
        x0 = boxes[..., 0, None] + torch.div(ix * bw, ow, rounding_mode="floor")
        x1 = boxes[..., 0, None] + torch.div((ix + 1) * bw + ow - 1, ow, rounding_mode="floor")
        integral = F.pad(images[:, 0].double().cumsum(-1).cumsum(-2), (1, 0, 1, 0))
        pitch = integral.shape[-1]
        integral = integral.flatten(1)
        values = integral.gather(1, (y1[..., :, None] * pitch + x1[..., None, :]).flatten(1))
        values -= integral.gather(1, (y0[..., :, None] * pitch + x1[..., None, :]).flatten(1))
        values -= integral.gather(1, (y1[..., :, None] * pitch + x0[..., None, :]).flatten(1))
        values += integral.gather(1, (y0[..., :, None] * pitch + x0[..., None, :]).flatten(1))
        area = (y1 - y0)[..., :, None] * (x1 - x0)[..., None, :]
        return (values.view(n, k, height, width) / area).float()

    def descriptor(self, patches):
        """Return normalized radial descriptors; constant patches occupy the lowest band."""
        patches = patches.reshape(-1, 1, self.size, self.size)
        mean = (patches * self.window).sum((-2, -1), keepdim=True) / self.window.sum()
        spectrum = torch.fft.rfft2((patches - mean) * self.window).abs().log1p()
        energy = torch.einsum("nchw,khw->nk", spectrum, self.bands)
        descriptors = energy / energy.sum(1, keepdim=True).clamp_min(1e-12)
        flat = (patches.amax((-2, -1)) - patches.amin((-2, -1)))[:, 0] <= 1e-6
        return torch.where(flat[:, None], descriptors.new_tensor([1, 0, 0, 0]), descriptors)

    @torch.no_grad()
    def __call__(self, images, targets):
        """Analyze final augmented images without per-target resize or FFT calls."""
        retention = images.new_ones((len(targets), len(self.strides)), dtype=torch.float32)
        if not len(targets):
            return retention
        self.window, self.bands = self.window.to(images.device), self.bands.to(images.device)
        gray = (images.float() * images.new_tensor([0.299, 0.587, 0.114])[None, :, None, None]).sum(1, keepdim=True)
        # Match the old Python coordinate arithmetic before integer crop rounding.
        targets = targets.to(device=images.device, dtype=torch.float64)
        limits = targets.new_tensor([images.shape[3], images.shape[2]])
        center, half = targets[:, 2:4] * limits, targets[:, 4:6] * limits / 2
        gt0, gt1 = (center - half).floor().clamp_min(0), (center + half).ceil().minimum(limits)
        ctx0 = (center - half * self.context).floor().clamp_min(0)
        ctx1 = (center + half * self.context).ceil().minimum(limits)
        wh = (ctx1 - ctx0).long()
        obj = torch.cat((gt0 - ctx0, gt1 - ctx0), 1).long()
        x0, y0, x1, y1 = obj.unbind(1)
        w, h = wh.unbind(1)
        zero = torch.zeros_like(w)
        regions = torch.stack(
            (
                obj,
                torch.stack((zero, zero, w, y0), 1),
                torch.stack((zero, y1, w, h), 1),
                torch.stack((zero, y0, x0, y1), 1),
                torch.stack((x1, y0, w, y1), 1),
            ),
            1,
        )
        valid = ((regions[..., 2:] - regions[..., :2]) >= 2).all(2)
        supported = valid[:, 0] & valid[:, 1:].any(1)
        regions = torch.where(valid[..., None], regions, regions.new_tensor([0, 0, 1, 1]))
        buckets = (2 ** wh.max(1)[0].clamp_min(16).float().log2().ceil()).long().clamp_max(max(images.shape[2:]))
        for side in buckets[supported].unique().tolist():
            group = ((buckets == side) & supported).nonzero(as_tuple=True)[0]
            # Bound padded image and FFT storage without creating a per-object kernel path.
            for ids in group.split(min(512, max(1, 8388608 // (side * side)))):
                start = ctx0[ids].float()
                rois = torch.cat((targets[ids, :1].float(), start, start + side), 1)
                # Native forward is deterministic and needs no backward. Unit bins reproduce pixel
                # crops and avoid torchvision's backward-oriented deterministic decomposition.
                crop = torch.ops.torchvision.roi_align(gray, rois, 1.0, side, side, 1, True)
                boxes = regions[ids]
                native_hw = wh[ids][:, [1, 0]]
                full = torch.cat((torch.zeros_like(wh[ids]), wh[ids]), 1)[:, None]
                patches = [self.area_pool(crop, boxes, (self.size, self.size))]
                position = torch.arange(side, device=images.device, dtype=torch.float32)[None, :] + 0.5
                for stride in self.strides:
                    sampled_hw = (native_hw.double() / stride).ceil().long().clamp_min(1)
                    storage = max(1, math.ceil(side / stride))
                    sampled = self.area_pool(crop, full, (storage, storage), sampled_hw)
                    source = position[:, :, None] * (sampled_hw.float() / native_hw)[:, None] - 0.5
                    source = source.clamp_min(0).minimum(sampled_hw[:, None] - 1)
                    gy, gx = (2 * (source + 0.5) / storage - 1).unbind(2)
                    grid = torch.stack((gx[:, None, :].expand(-1, side, -1), gy[:, :, None].expand(-1, -1, side)), -1)
                    restored = F.grid_sample(sampled, grid, mode="bilinear", padding_mode="border", align_corners=False)
                    patches.append(self.area_pool(restored, boxes, (self.size, self.size)))
                descriptors = self.descriptor(torch.stack(patches, 1)).view(len(ids), len(self.strides) + 1, 5, 4)
                obj = descriptors[:, :, 0]
                context_valid = valid[ids, 1:]
                ctx = (descriptors[:, :, 1:] * context_valid[:, None, :, None]).sum(2)
                ctx /= context_valid.sum(1)[:, None, None]
                midpoint = (obj + ctx) / 2
                distance = 0.5 * (
                    (obj * (obj.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(2)
                    + (ctx * (ctx.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(2)
                )
                baseline = distance[:, :1]
                scores = (distance[:, 1:] / baseline.clamp_min(self.min_contrast)).clamp(0, 1)
                retention[ids] = torch.where(baseline >= self.min_contrast, scores, torch.ones_like(scores))
        return retention
