# torchbox

A PyTorch + OpenCV project.

## Modules

- `torchbox.cli` — Command line entry point (`torchbox` / `python3 -m torchbox`). Dispatches
  to the `hello`, `mnist`, `yolo`, `live-yolo`, and `finance` subcommands, forwarding any
  extra arguments to the corresponding module.
- `torchbox.mnist` — A standard MNIST convolutional network adapted from the PyTorch
  examples. Supports training, evaluation, checkpoint load/save, and inference on image
  files or webcam frames.
- `torchbox.yolo` — A simplified YOLO-like object detector trained on synthetic geometric
  shapes (Circle, Rectangle, Triangle). Supports training, weight load/save, and inference
  on image files or single webcam frames.
- `torchbox.live_yolo` — Real-time YOLO shape detection on a live video stream using
  OpenCV. Loads trained weights (default `models/yolo.pt`), annotates each captured frame,
  and displays the result in a window (press `q` to quit).
- `torchbox.finance` — A wide-window RNN model for stock price boost prediction. Trains on
  live data fetched from the Yahoo Finance API (e.g. `--ticker AAPL`) or on synthetic data
  (`--synthetic`), with model checkpoint load/save.

## Usage

```bash
python3 -m torchbox --name Ryan
# or, once installed:
torchbox --name Ryan
```

Train the MNIST model, optionally starting from an existing checkpoint:

```bash
python3 -m torchbox mnist --train --save-model models/model.pt
python3 -m torchbox mnist --train --load-model models/model.pt --save-model models/model.pt
```

Run inference on an image. Inference is the default mode and uses
`models/model.pt` when `--load-model` is omitted:

```bash
python3 -m torchbox mnist --input frame.png
python3 -m torchbox mnist --infer --load-model models/weights.pt --input frame.png
```

## Dev container configuration

The project ships a dev container (see [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json))
with PyTorch, OpenCV, and the Python tooling preinstalled. For headless work (training,
tests, file-based inference) the defaults are enough. To test GUI/webcam programs such as
`torchbox.live_yolo`, which open an OpenCV window and read from a camera, a few host
resources must be forwarded into the container:

### Webcam access

Uncomment the video device entries in `runArgs` that match the camera(s) you want to
expose (check `ls /dev/video*` on the host):

```jsonc
"runArgs": [
	"--device=/dev/video0",
	// "--device=/dev/video1",
	...
]
```

Note that `--device=/dev/videoN` fails at container startup if the device is absent on
the host. To grant access to whatever video4l2 devices happen to be present (including
hot-plugged cameras) without failing when none exist, use a device cgroup rule plus a
`/dev` bind mount instead:

```jsonc
"runArgs": [
	"--device-cgroup-rule=c 81:* rmw",  // char major 81 = video4linux
	"-v", "/dev:/dev",
	...
]
```

This is more flexible but less isolated, since all of `/dev` is visible inside the
container (the cgroup rule still limits device access to video devices).

The camera index passed to `--cam-input` corresponds to the forwarded device.

### GUI display (OpenCV windows)

Display forwarding is already configured and relies on:

- `--net=host` and `--ipc=host` in `runArgs` — local socket access and MIT-SHM shared
  memory for X11.
- `--device=/dev/dri:/dev/dri` — GPU hardware acceleration for rendering.
- `containerEnv` — passes `DISPLAY`, `WAYLAND_DISPLAY`, `XDG_RUNTIME_DIR`, and points
  `XAUTHORITY` at the mounted credentials.
- `mounts` — binds the host's X11 socket (`/tmp/.X11-unix`), `~/.Xauthority`, and the
  Wayland socket into the container.

These require an X11 or Wayland session on the host. If the window fails to open, verify
that `DISPLAY` is set inside the container and that the host allows local connections
(e.g. `xhost +local:` on the host).

### NVIDIA GPU (optional)

To run inference/training on an NVIDIA GPU, install `nvidia-container-toolkit` on the
host and add to `runArgs`:

```jsonc
"runArgs": ["--gpus", "all", ...]
```

`live_yolo` automatically uses CUDA when available, falling back to CPU otherwise.

After editing `devcontainer.json`, rebuild the container ("Dev Containers: Rebuild
Container" in VS Code), then run:

```bash
python3 -m torchbox live-yolo --cam-input 0 --model-path models/yolo.pt
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
```
