"""Focused regression checks for frozen Hard-SRLA assignment and batched sampling."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from utils.loss import ComputeLoss
from utils.srla import SRLAAnalyzer


def make_loss():
    model = torch.nn.Module()
    model.register_parameter("dummy", torch.nn.Parameter(torch.zeros(1)))
    model.hyp = {
        "cls_pw": 1,
        "obj_pw": 1,
        "fl_gamma": 0,
        "anchor_t": 4,
        "srla_thr": 0.5,
        "box": 0.05,
        "obj": 1,
        "cls": 0.5,
    }
    model.model = [SimpleNamespace(na=1, nc=2, nl=3, anchors=torch.ones(3, 1, 2))]
    predictions = [torch.zeros(1, 1, size, size, 7, requires_grad=True) for size in (8, 4, 2)]
    targets = torch.tensor([[0, 1, 0.43, 0.37, 0.25, 0.25]])
    return ComputeLoss(model), predictions, targets


def test_neutral_retention_preserves_loss_and_empty_targets():
    loss, predictions, targets = make_loss()
    for target in (targets, targets[:0]):
        baseline, items = loss(predictions, target)
        gated, gated_items = loss(predictions, target, torch.ones(len(target), 3))
        assert torch.equal(baseline, gated)
        assert torch.equal(items, gated_items)
        gated.backward()
        assert all(torch.isfinite(p.grad).all() for p in predictions)


def test_strict_threshold_fallback_and_compatible_level():
    loss, predictions, targets = make_loss()
    # Every score equal to the threshold fails the strict gate; fallback restores P3 only.
    output = loss.build_targets(predictions, targets, torch.full((1, 3), 0.5))
    assert [len(indices[0]) > 0 for indices in output[2]] == [True, False, False]
    assert torch.equal(loss.srla_counts, torch.tensor([[1, 1, 1], [1, 0, 0]]))
    output = loss.build_targets(predictions, targets, torch.tensor([[0.49, 0.51, 0.2]]))
    assert [len(indices[0]) > 0 for indices in output[2]] == [False, True, False]
    targets[:, 4:6] = 0.00001  # Originally unmatched GT must remain unmatched.
    output = loss.build_targets(predictions, targets, torch.ones(1, 3))
    assert all(len(indices[0]) == 0 for indices in output[2])


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_batched_area_pool_matches_native_and_flat_retention(device):
    torch.manual_seed(0)
    images = torch.rand(2, 1, 16, 16, device=device)
    boxes = torch.tensor([[[1, 2, 12, 15]], [[0, 1, 8, 10]]], device=device)
    pooled = SRLAAnalyzer.area_pool(images, boxes, (5, 6))
    reference = torch.cat(
        [
            F.interpolate(images[:1, :, 2:15, 1:12], (5, 6), mode="area"),
            F.interpolate(images[1:, :, 1:10, :8], (5, 6), mode="area"),
        ]
    )
    torch.testing.assert_close(pooled, reference, atol=2e-7, rtol=1e-6)
    analyzer = SRLAAnalyzer([8, 16, 32])
    flat = torch.ones(1, 3, 64, 64, device=device, requires_grad=True)
    targets = torch.tensor([[0, 0, 0.5, 0.5, 0.2, 0.2], [0, 1, 0.02, 0.02, 0.01, 0.01]], device=device)
    retention = analyzer(flat, targets)
    assert torch.equal(retention, torch.ones(2, 3, device=device))
    assert not retention.requires_grad
    assert analyzer(flat, targets[:0]).shape == (0, 3)
