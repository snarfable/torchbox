"""A standard MNIST convolutional network, adapted from the PyTorch examples.

Train and save weights::

    python3 -m torchbox.mnist --epochs 5 --save-model mnist_cnn.pt

Load saved weights and evaluate without training::

    python3 -m torchbox.mnist --load-model mnist_cnn.pt --epochs 0

Run inference for an image (inference is the default mode)::

    python3 -m torchbox.mnist --load-model models/model.pt --input frame.png

Load weights, keep training, then re-save::

    python3 -m torchbox.mnist --load-model mnist_cnn.pt --epochs 2 --save-model mnist_cnn.pt

All of the above also work via the torchbox CLI, e.g. ``torchbox mnist --epochs 5``.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class Net(nn.Module):
    """A small CNN: two conv layers, dropout, two fully-connected layers."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    epoch: int,
    log_interval: int = 100,
) -> None:
    """Run one training epoch."""
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            seen = batch_idx * len(data)
            total = len(loader.dataset)
            pct = 100.0 * batch_idx / len(loader)
            print(f"Train epoch {epoch} [{seen}/{total} ({pct:.0f}%)]  loss: {loss.item():.6f}")


def test(model: nn.Module, device: torch.device, loader: DataLoader) -> float:
    """Evaluate on the test set. Returns accuracy in [0, 1]."""
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    total = len(loader.dataset)
    test_loss /= total
    accuracy = correct / total
    print(f"Test set: avg loss: {test_loss:.4f}, accuracy: {correct}/{total} ({100.0 * accuracy:.2f}%)")
    return accuracy


def infer(
    model: nn.Module,
    device: torch.device,
    input_path: str,
    transform: transforms.Compose,
) -> int:
    """Predict the digit in one image and print the result."""
    image = Image.open(input_path).convert("L")
    data = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(data).argmax(dim=1).item()
    print(f"Inference for {input_path}: {prediction}")
    return prediction


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train or run inference with an MNIST CNN.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--train", action="store_true", help="train the model")
    mode.add_argument("--infer", action="store_true", help="run inference (the default)")
    parser.add_argument("--batch-size", type=int, default=64, help="training batch size")
    parser.add_argument("--test-batch-size", type=int, default=1000, help="test batch size")
    parser.add_argument("--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1.0, help="learning rate")
    parser.add_argument("--gamma", type=float, default=0.7, help="learning rate step gamma")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument("--no-cuda", action="store_true", help="disable CUDA even if available")
    parser.add_argument("--no-mps", action="store_true", help="disable macOS MPS even if available")
    parser.add_argument("--data-dir", default="./data", help="where to store the MNIST dataset")
    parser.add_argument(
        "--save-model",
        nargs="?",
        const="mnist_cnn.pt",
        default=None,
        metavar="PATH",
        help="save trained weights to PATH when done (default PATH: mnist_cnn.pt)",
    )
    parser.add_argument(
        "--load-model",
        default=None,
        metavar="PATH",
        help="load weights from PATH before training or inference",
    )
    parser.add_argument("--input", metavar="PATH", help="image to classify in inference mode")
    return parser


def select_device(args: argparse.Namespace) -> torch.device:
    if not args.no_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if not args.no_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    training = args.train or args.save_model is not None
    if args.infer and args.save_model is not None:
        build_arg_parser().error("--save-model implies --train and cannot be used with --infer")
    if not training:
        model_path = args.load_model or "models/model.pt"
        if not args.input:
            build_arg_parser().error("inference requires --input PATH")
    else:
        model_path = args.load_model

    torch.manual_seed(args.seed)
    device = select_device(args)
    print(f"Using device: {device}")

    # MNIST's standard normalization (dataset mean/std).
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    inference_transform = transforms.Compose(
        [
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    model = Net().to(device)

    if model_path:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {model_path}")

    if not training:
        infer(model, device, args.input, inference_transform)
        return 0

    train_dataset = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(args.data_dir, train=False, transform=transform)

    use_cuda = device.type == "cuda"
    loader_kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **loader_kwargs)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.epochs < 1:
        test(model, device, test_loader)
    else:
        for epoch in range(1, args.epochs + 1):
            train(model, device, train_loader, optimizer, epoch)
            test(model, device, test_loader)
            scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), args.save_model)
        print(f"Saved weights to {args.save_model}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
