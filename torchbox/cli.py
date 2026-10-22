"""Command line interface for torchbox."""

import argparse

from torchbox import __version__


def main(argv: list[str] | None = None) -> int:
    """Run the torchbox CLI. Returns a process exit code."""
    parser = argparse.ArgumentParser(
        prog="torchbox",
        description="A PyTorch + OpenCV project.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")

    hello = subparsers.add_parser("hello", help="print a greeting")
    hello.add_argument(
        "-n",
        "--name",
        default="world",
        help="who to greet (default: world)",
    )

    # `torchbox mnist ...` — forwards remaining args to the MNIST trainer.
    subparsers.add_parser(
        "mnist",
        help="train the sample MNIST model",
        add_help=False,
    )

    # `torchbox yolo ...` — forwards remaining args to the YOLO detector.
    subparsers.add_parser(
        "yolo",
        help="train or run inference with the YOLO shape detector",
        add_help=False,
    )

    subparsers.add_parser(
        "live-yolo",
        help="run live YOLO detection from a video source",
    )

    # `torchbox finance ...` — forwards remaining args to the finance trainer/predictor.
    subparsers.add_parser(
        "finance",
        help="train or run inference with the wide-window RNN stock prediction model",
        add_help=False,
    )

    args, extra = parser.parse_known_args(argv)

    if args.command == "mnist":
        from torchbox.mnist import main as mnist_main

        return mnist_main(extra)

    if args.command == "yolo":
        from torchbox.yolo import main as yolo_main

        return yolo_main(extra)

    if args.command == "live-yolo":
        from torchbox.live_yolo import main as live_yolo_main

        live_yolo_main()
        return 0

    if args.command == "finance":
        from torchbox.finance import main as finance_main

        return finance_main(extra)

    if args.command == "hello":
        print(f"Hello, {args.name}! torchbox {__version__} is ready.")
        return 0

    # No subcommand: default greeting.
    print(f"Hello, world! torchbox {__version__} is ready.")
    return 0
