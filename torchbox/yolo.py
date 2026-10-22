"""A simplified YOLO-like object detector for synthetic geometric shapes.

Train and save weights:

    python3 -m torchbox.yolo --epochs 10 --save-model models/yolo.pt

Load saved weights and run inference on an image (visualizing output to a file):

    python3 -m torchbox.yolo --load-model models/yolo.pt --input share/images/frame.png

Run inference on a webcam frame:

    python3 -m torchbox.yolo --load-model models/yolo.pt --cam-input 0

Integrates with the torchbox CLI:

    torchbox yolo --epochs 10 --save-model models/yolo.pt
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Class definitions for the shapes
CLASSES = ["Circle", "Rectangle", "Triangle"]
NUM_CLASSES = len(CLASSES)
S = 4  # Grid size (S x S)


def generate_synthetic_image(img_size: int = 128, grid_size: int = 4) -> tuple[Image.Image, np.ndarray]:
    """Generates a single synthetic image with 1 to 3 non-overlapping shapes.

    Returns:
        image: PIL.Image of shape (img_size, img_size, 3)
        target: numpy array of shape (grid_size, grid_size, 8)
                [x_cell, y_cell, w_norm, h_norm, confidence, p_circle, p_rectangle, p_triangle]
    """
    image = Image.new("RGB", (img_size, img_size), "black")
    draw = ImageDraw.Draw(image)
    target = np.zeros((grid_size, grid_size, 5 + NUM_CLASSES), dtype=np.float32)

    cell_size = img_size / grid_size
    num_shapes = random.randint(1, 3)

    # Select unique grid cells
    available_cells = [(j, i) for j in range(grid_size) for i in range(grid_size)]
    random.shuffle(available_cells)
    selected_cells = available_cells[:num_shapes]

    for j, i in selected_cells:
        # Determine cell center in pixel coordinates
        cx_pixel = (i + 0.5) * cell_size
        cy_pixel = (j + 0.5) * cell_size

        # Add small random offset so center is guaranteed to stay within the cell
        max_offset = cell_size * 0.25
        offset_x = random.uniform(-max_offset, max_offset)
        offset_y = random.uniform(-max_offset, max_offset)

        cx = cx_pixel + offset_x
        cy = cy_pixel + offset_y

        # Choose shape type: 0 = Circle, 1 = Rectangle, 2 = Triangle
        shape_type = random.randint(0, 2)

        # Choose random size relative to cell size
        size = random.uniform(cell_size * 0.4, cell_size * 0.8)
        half_size = size / 2.0

        xmin = cx - half_size
        ymin = cy - half_size
        xmax = cx + half_size
        ymax = cy + half_size

        # Choose random bright color
        color = tuple(random.randint(100, 255) for _ in range(3))

        # Draw the shape
        if shape_type == 0:
            draw.ellipse([xmin, ymin, xmax, ymax], fill=color)
        elif shape_type == 1:
            draw.rectangle([xmin, ymin, xmax, ymax], fill=color)
        elif shape_type == 2:
            # Triangle: pointing up
            v1 = (cx, ymin)
            v2 = (xmin, ymax)
            v3 = (xmax, ymax)
            draw.polygon([v1, v2, v3], fill=color)

        # Normalize coordinates relative to full image
        x_norm = cx / img_size
        y_norm = cy / img_size
        w_norm = size / img_size
        h_norm = size / img_size

        # Coordinate within the specific grid cell (0..1)
        x_cell = x_norm * grid_size - i
        y_cell = y_norm * grid_size - j

        # Store target vector
        target[j, i, 0] = x_cell
        target[j, i, 1] = y_cell
        target[j, i, 2] = w_norm
        target[j, i, 3] = h_norm
        target[j, i, 4] = 1.0  # objectness confidence
        target[j, i, 5 + shape_type] = 1.0  # one-hot class representation

    return image, target


class SyntheticShapesDataset(Dataset):
    """Dataset producing reproducible synthetic shape images and YOLO bounding box targets."""

    def __init__(
        self,
        num_samples: int = 500,
        img_size: int = 128,
        grid_size: int = 4,
        transform: transforms.Compose | None = None,
    ) -> None:
        self.num_samples = num_samples
        self.img_size = img_size
        self.grid_size = grid_size
        self.transform = transform

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Local seeding to ensure reproducibility of dataset samples
        prev_state = random.getstate()
        prev_np_state = np.random.get_state()
        
        random.seed(idx + 100000)
        np.random.seed(idx + 100000)
        
        image, target = generate_synthetic_image(self.img_size, self.grid_size)
        
        random.setstate(prev_state)
        np.random.set_state(prev_np_state)

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        return image_tensor, torch.tensor(target, dtype=torch.float32)


class YoloNet(nn.Module):
    """A lightweight fully convolutional YOLO network for S x S grid object detection."""

    def __init__(self, grid_size: int = 4, num_classes: int = 3) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.num_classes = num_classes

        # input: 3 x 128 x 128
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Output layer maps to (5 + num_classes) predictions per cell
        # Grid size S=4 is reached via 5 pooling layers (128 -> 64 -> 32 -> 16 -> 8 -> 4)
        self.out_conv = nn.Conv2d(256, 5 + num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # 64
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # 32
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # 16
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)  # 8
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, 2)  # 4

        out = self.out_conv(x)  # shape: (batch_size, 5 + num_classes, grid_size, grid_size)
        out = out.permute(0, 2, 3, 1)  # shape: (batch_size, grid_size, grid_size, 5 + num_classes)

        # Coordinate bounding boxes and object confidence are scaled between 0 and 1 via sigmoid
        coords_conf = torch.sigmoid(out[..., :5])
        # Class probabilities are predicted as log probabilities
        classes = F.log_softmax(out[..., 5:], dim=-1)

        return torch.cat([coords_conf, classes], dim=-1)


class YoloLoss(nn.Module):
    """Custom YOLO loss function with coordinate, object confidence, and class loss components."""

    def __init__(self, coord_coef: float = 5.0, noobj_coef: float = 0.5) -> None:
        super().__init__()
        self.coord_coef = coord_coef
        self.noobj_coef = noobj_coef

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # predictions/targets shape: (batch_size, S, S, 5 + num_classes)
        # element layout: [x_cell, y_cell, w_norm, h_norm, confidence, p_class...]
        obj_mask = targets[..., 4] == 1.0
        noobj_mask = targets[..., 4] == 0.0

        # 1. Coordinate Loss (MSE for x, y, w, h only where objects exist)
        pred_coords = predictions[..., :4][obj_mask]
        target_coords = targets[..., :4][obj_mask]
        if pred_coords.numel() > 0:
            loss_coord = self.coord_coef * F.mse_loss(pred_coords, target_coords, reduction="sum")
        else:
            loss_coord = 0.0

        # 2. Confidence Loss (Object)
        pred_conf_obj = predictions[..., 4][obj_mask]
        target_conf_obj = targets[..., 4][obj_mask]
        if pred_conf_obj.numel() > 0:
            loss_conf_obj = F.mse_loss(pred_conf_obj, target_conf_obj, reduction="sum")
        else:
            loss_conf_obj = 0.0

        # 3. Confidence Loss (No Object, down-weighted)
        pred_conf_noobj = predictions[..., 4][noobj_mask]
        target_conf_noobj = targets[..., 4][noobj_mask]
        loss_conf_noobj = self.noobj_coef * F.mse_loss(pred_conf_noobj, target_conf_noobj, reduction="sum")

        # 4. Classification Loss (NLL loss for log-probabilities)
        pred_classes = predictions[..., 5:][obj_mask]
        target_classes = targets[..., 5:][obj_mask].argmax(dim=-1)
        if pred_classes.numel() > 0:
            loss_class = F.nll_loss(pred_classes, target_classes, reduction="sum")
        else:
            loss_class = 0.0

        batch_size = predictions.size(0)
        return (loss_coord + loss_conf_obj + loss_conf_noobj + loss_class) / batch_size


def train(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epoch: int,
    log_interval: int = 10,
) -> None:
    """Runs a single training epoch."""
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            seen = batch_idx * len(data)
            total = len(loader.dataset)
            pct = 100.0 * batch_idx / len(loader)
            print(f"Train Epoch: {epoch} [{seen}/{total} ({pct:.0f}%)]  Loss: {loss.item():.6f}")


def evaluate(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """Evaluates the model on validation data. Returns average loss."""
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item() * data.size(0)

    avg_loss = val_loss / len(loader.dataset)
    print(f"Validation set: Average loss: {avg_loss:.4f}")
    return avg_loss


def nms(boxes: list[list[float]], scores: list[float], iou_threshold: float = 0.5) -> list[int]:
    """Applies Non-Maximum Suppression (NMS) to eliminate overlapping bounding boxes."""
    if not boxes:
        return []

    box_arr = np.array(boxes)
    score_arr = np.array(scores)

    x1 = box_arr[:, 0]
    y1 = box_arr[:, 1]
    x2 = box_arr[:, 2]
    y2 = box_arr[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = score_arr.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


def decode_predictions(
    predictions: np.ndarray,
    img_size: int = 128,
    conf_threshold: float = 0.5,
    grid_size: int = 4,
) -> tuple[list[list[float]], list[float], list[int]]:
    """Decodes S x S grid predictions into boxes, confidence scores, and class IDs."""
    boxes = []
    scores = []
    class_ids = []

    for j in range(grid_size):
        for i in range(grid_size):
            cell = predictions[j, i]
            conf = cell[4]
            if conf >= conf_threshold:
                x_cell, y_cell = cell[0], cell[1]
                w_norm, h_norm = cell[2], cell[3]

                # Convert to absolute coordinates on img_size
                cx = ((i + x_cell) / grid_size) * img_size
                cy = ((j + y_cell) / grid_size) * img_size
                w = w_norm * img_size
                h = h_norm * img_size

                xmin = cx - w / 2.0
                ymin = cy - h / 2.0
                xmax = cx + w / 2.0
                ymax = cy + h / 2.0

                # Clamp values to image bounds
                xmin = max(0.0, min(float(xmin), float(img_size)))
                ymin = max(0.0, min(float(ymin), float(img_size)))
                xmax = max(0.0, min(float(xmax), float(img_size)))
                ymax = max(0.0, min(float(ymax), float(img_size)))

                class_id = int(np.argmax(cell[5:]))

                boxes.append([xmin, ymin, xmax, ymax])
                scores.append(float(conf))
                class_ids.append(class_id)

    return boxes, scores, class_ids


def draw_detections(
    image_np: np.ndarray,
    boxes: list[list[float]],
    scores: list[float],
    class_ids: list[int],
) -> np.ndarray:
    """Draws bounding boxes and labels onto a numpy BGR image."""
    import cv2

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = f"{CLASSES[class_id]} {score:.2f}"

        # BGR: Green for Circle, Blue for Rectangle, Red for Triangle
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255)]
        color = colors[class_id % len(colors)]

        # Draw box
        cv2.rectangle(image_np, (x1, y1), (x2, y2), color, 2)

        # Text size for background box
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(image_np, (x1, y1 - h - 4), (x1 + w, y1), color, -1)

        # Draw text label
        cv2.putText(
            image_np,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return image_np


def predict(
    model: nn.Module,
    device: torch.device,
    image: Image.Image,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> tuple[list[list[float]], list[float], list[int]]:
    """Predicts bounding boxes, confidences, and labels for a single PIL image."""
    orig_w, orig_h = image.size

    # Prepare image for model input
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    data = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(data).squeeze(0).cpu().numpy()  # (S, S, 5 + C)

    # Decode predictions in 128x128 space
    boxes_128, scores, class_ids = decode_predictions(output, img_size=128, conf_threshold=conf_threshold)

    # Apply Non-Maximum Suppression
    keep_indices = nms(boxes_128, scores, iou_threshold=iou_threshold)
    boxes_filtered = [boxes_128[idx] for idx in keep_indices]
    scores_filtered = [scores[idx] for idx in keep_indices]
    class_ids_filtered = [class_ids[idx] for idx in keep_indices]

    # Map box coordinates back to original high-res image dimensions
    boxes_orig = []
    for box in boxes_filtered:
        x1 = (box[0] / 128.0) * orig_w
        y1 = (box[1] / 128.0) * orig_h
        x2 = (box[2] / 128.0) * orig_w
        y2 = (box[3] / 128.0) * orig_h
        boxes_orig.append([x1, y1, x2, y2])

    return boxes_orig, scores_filtered, class_ids_filtered


def infer(
    model: nn.Module,
    device: torch.device,
    input_path: str,
    output_path: str = "output_yolo.png",
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> None:
    """Predicts objects in a file, prints results, and saves the annotated image."""
    import cv2

    image = Image.open(input_path).convert("RGB")
    boxes, scores, class_ids = predict(
        model,
        device,
        image,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    print(f"Inference for {input_path}: found {len(boxes)} object(s).")
    for idx, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        box_int = [int(v) for v in box]
        print(f"  Object {idx}: {CLASSES[class_id]} (Confidence: {score:.2f}), Box: {box_int}")

    # Annotate original high-res image
    image_cv = cv2.imread(input_path)
    if image_cv is None:
        # Fallback if cv2 can't read the file directly (e.g., custom format)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    annotated_cv = draw_detections(image_cv, boxes, scores, class_ids)
    cv2.imwrite(output_path, annotated_cv)
    print(f"Annotated output saved to {output_path}")

def infer_in_memory(
    model: nn.Module,
    device: torch.device,
    image: np.ndarray,
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> np.ndarray:
    """Predicts objects in a file, prints results, and saves the annotated image."""
    import cv2

    image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    boxes, scores, class_ids = predict(
        model,
        device,
        image_pil,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    #print(f"Inference for image: found {len(boxes)} object(s).")
    #for idx, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
    #    box_int = [int(v) for v in box]
    #    print(f"  Object {idx}: {CLASSES[class_id]} (Confidence: {score:.2f}), Box: {box_int}")

    return draw_detections(image, boxes, scores, class_ids)

def infer_from_camera(
    model: nn.Module,
    device: torch.device,
    cam_index: int,
    output_path: str = "output_yolo.png",
    conf_threshold: float = 0.5,
    iou_threshold: float = 0.5,
) -> None:
    """Captures one frame from a webcam, detects shapes, and saves the result."""
    import cv2

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {cam_index}")
    try:
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Failed to capture a frame from webcam index {cam_index}")

    # Convert frame to PIL image for the prediction pipeline
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    boxes, scores, class_ids = predict(
        model,
        device,
        image,
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )

    print(f"Webcam capture from index {cam_index}: found {len(boxes)} object(s).")
    for idx, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
        box_int = [int(v) for v in box]
        print(f"  Object {idx}: {CLASSES[class_id]} (Confidence: {score:.2f}), Box: {box_int}")

    annotated_frame = draw_detections(frame, boxes, scores, class_ids)
    cv2.imwrite(output_path, annotated_frame)
    print(f"Annotated frame saved to {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds the argument parser for YOLO model operations."""
    parser = argparse.ArgumentParser(description="Train or run inference with a custom YOLO shape detector.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="train the model")
    mode.add_argument("--infer", action="store_true", help="run inference (the default)")

    parser.add_argument("--batch-size", type=int, default=32, help="training batch size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs to train")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--no-cuda", action="store_true", help="disable CUDA even if available")
    parser.add_argument("--no-mps", action="store_true", help="disable macOS MPS even if available")
    parser.add_argument("--conf-threshold", type=float, default=0.5, help="confidence threshold for detection")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for NMS")

    parser.add_argument(
        "--save-model",
        nargs="?",
        const="models/yolo.pt",
        default=None,
        metavar="PATH",
        help="save trained weights to PATH when done (default PATH: models/yolo.pt)",
    )
    parser.add_argument(
        "--load-model",
        default=None,
        metavar="PATH",
        help="load weights from PATH before training or inference",
    )
    parser.add_argument(
        "--output",
        default="output_yolo.png",
        metavar="PATH",
        help="output path to save detection visualization image",
    )

    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", metavar="PATH", help="image file to classify in inference mode")
    source.add_argument(
        "--cam-input",
        type=int,
        metavar="INDEX",
        help="capture one frame from webcam at INDEX to classify (e.g. --cam-input 0)",
    )
    return parser


def select_device(args: argparse.Namespace) -> torch.device:
    """Selects the best available device based on args and hardware capabilities."""
    if not args.no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if not args.no_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    """The CLI entry point for the YOLO module."""
    args = build_arg_parser().parse_args(argv)
    training = args.train or args.save_model is not None

    if args.infer and args.save_model is not None:
        build_arg_parser().error("--save-model implies --train and cannot be used with --infer")

    if not training:
        model_path = args.load_model or "models/yolo.pt"
        if not args.input and args.cam_input is None:
            build_arg_parser().error("Inference requires --input PATH or --cam-input INDEX")
    else:
        model_path = args.load_model

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args)
    print(f"Using device: {device}")

    model = YoloNet(grid_size=S, num_classes=NUM_CLASSES).to(device)

    if model_path:
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            print(f"Loaded weights from {model_path}")
        elif not training:
            # If doing inference and the explicit models/yolo.pt or load-model does not exist, fail gracefully
            print(f"Error: Model weights not found at {model_path}. Please train the model first.")
            return 1

    if not training:
        if args.cam_input is not None:
            infer_from_camera(
                model,
                device,
                args.cam_input,
                output_path=args.output,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
            )
        else:
            infer(
                model,
                device,
                args.input,
                output_path=args.output,
                conf_threshold=args.conf_threshold,
                iou_threshold=args.iou_threshold,
            )
        return 0

    # Ensure output parent directory exists for saving weights
    if args.save_model:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_model)), exist_ok=True)

    print("Generating synthetic dataset...")
    train_dataset = SyntheticShapesDataset(num_samples=500, img_size=128, grid_size=S)
    val_dataset = SyntheticShapesDataset(num_samples=100, img_size=128, grid_size=S)

    use_cuda = device.type == "cuda"
    loader_kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = YoloLoss()

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        evaluate(model, device, val_loader, criterion)

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"Saved weights to {args.save_model}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
