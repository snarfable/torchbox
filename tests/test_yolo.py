"""Unit tests for the YOLO shape detection module."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from torchbox.yolo import (
    SyntheticShapesDataset,
    YoloLoss,
    YoloNet,
    decode_predictions,
    nms,
)


def test_synthetic_shapes_dataset() -> None:
    """Verifies that the synthetic shapes dataset generates samples with expected shapes and values."""
    img_size = 128
    grid_size = 4
    num_samples = 5
    dataset = SyntheticShapesDataset(num_samples=num_samples, img_size=img_size, grid_size=grid_size)

    assert len(dataset) == num_samples

    # Inspect a single sample
    image, target = dataset[0]

    # Verify image dimensions: (channels, height, width)
    assert isinstance(image, torch.Tensor)
    assert image.shape == (3, img_size, img_size)
    assert image.dtype == torch.float32
    assert image.min() >= 0.0
    assert image.max() <= 1.0

    # Verify target dimensions: (grid_size, grid_size, 8)
    assert isinstance(target, torch.Tensor)
    assert target.shape == (grid_size, grid_size, 8)
    assert target.dtype == torch.float32

    # Check that coordinate boundaries are valid where objects are defined
    obj_mask = target[..., 4] == 1.0
    if obj_mask.any():
        coords = target[..., :4][obj_mask]
        # x, y in grid coordinates should be in [0, 1]
        assert (coords[:, :2] >= 0.0).all() and (coords[:, :2] <= 1.0).all()
        # w, h in normalized image coordinates should be in [0, 1]
        assert (coords[:, 2:] >= 0.0).all() and (coords[:, 2:] <= 1.0).all()

        # Class probabilities should sum to 1.0 for cells containing objects
        classes = target[..., 5:][obj_mask]
        assert torch.allclose(classes.sum(dim=-1), torch.ones_like(classes.sum(dim=-1)))


def test_yolo_net_forward() -> None:
    """Verifies that YoloNet forward pass produces predictions of the expected shape and values."""
    grid_size = 4
    num_classes = 3
    model = YoloNet(grid_size=grid_size, num_classes=num_classes)

    # Dummy batch of size 2
    x = torch.randn(2, 3, 128, 128)
    out = model(x)

    # Expected shape: (batch_size, grid_size, grid_size, 5 + num_classes)
    assert out.shape == (2, grid_size, grid_size, 8)

    # Coordinates and confidence should be in [0, 1] (due to sigmoid)
    assert (out[..., :5] >= 0.0).all() and (out[..., :5] <= 1.0).all()

    # Classes log-probabilities should be non-positive
    assert (out[..., 5:] <= 0.0).all()


def test_yolo_loss() -> None:
    """Verifies that YoloLoss computes a valid scalar loss."""
    grid_size = 4
    num_classes = 3
    criterion = YoloLoss()

    # Mock predictions: shape (2, 4, 4, 8)
    # Scaled properly: coords/conf in [0, 1], classes in negative range (log_softmax)
    preds = torch.cat([
        torch.sigmoid(torch.randn(2, grid_size, grid_size, 5)),
        torch.log_softmax(torch.randn(2, grid_size, grid_size, num_classes), dim=-1)
    ], dim=-1)

    # Mock targets: shape (2, 4, 4, 8)
    # One object in batch 0, cell (1, 1). No objects in batch 1.
    targets = torch.zeros(2, grid_size, grid_size, 5 + num_classes)
    targets[0, 1, 1, :5] = torch.tensor([0.5, 0.5, 0.2, 0.2, 1.0])
    targets[0, 1, 1, 5] = 1.0  # circle class

    loss = criterion(preds, targets)

    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar
    assert loss.item() >= 0.0


def test_nms() -> None:
    """Verifies that NMS correctly filters overlapping boxes."""
    # Two heavily overlapping boxes, one has a much higher score
    boxes = [
        [10.0, 10.0, 50.0, 50.0],
        [12.0, 12.0, 48.0, 48.0],  # high overlap
        [80.0, 80.0, 110.0, 110.0], # no overlap
    ]
    scores = [0.9, 0.4, 0.8]

    # With IoU threshold 0.5, the second box should be suppressed, keeping boxes 0 and 2.
    keep = nms(boxes, scores, iou_threshold=0.5)

    assert keep == [0, 2]


def test_decode_predictions() -> None:
    """Verifies that decode_predictions correctly decodes grid coordinates to pixel space."""
    grid_size = 4
    img_size = 128
    
    # Create empty mock predictions
    preds = np.zeros((grid_size, grid_size, 8), dtype=np.float32)
    # Add an object at grid cell (1, 2) i.e. j=1, i=2
    # [x_cell, y_cell, w_norm, h_norm, confidence, log_prob_circle, log_prob_rect, log_prob_tri]
    preds[1, 2] = [0.5, 0.5, 0.25, 0.25, 0.9, 0.0, -10.0, -10.0]

    boxes, scores, class_ids = decode_predictions(
        preds,
        img_size=img_size,
        conf_threshold=0.5,
        grid_size=grid_size,
    )

    assert len(boxes) == 1
    assert len(scores) == 1
    assert len(class_ids) == 1

    # Grid cell size = 128 / 4 = 32
    # cell (1, 2) corresponds to:
    # x center = (2 + 0.5) * 32 = 80
    # y center = (1 + 0.5) * 32 = 48
    # width = 0.25 * 128 = 32
    # height = 0.25 * 128 = 32
    # xmin = 80 - 16 = 64
    # ymin = 48 - 16 = 32
    # xmax = 80 + 16 = 96
    # ymax = 48 + 16 = 64
    assert pytest.approx(boxes[0][0]) == 64.0
    assert pytest.approx(boxes[0][1]) == 32.0
    assert pytest.approx(boxes[0][2]) == 96.0
    assert pytest.approx(boxes[0][3]) == 64.0
    assert pytest.approx(scores[0]) == 0.9
    assert class_ids[0] == 0  # Circle
