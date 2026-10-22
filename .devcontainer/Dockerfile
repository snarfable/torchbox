# Debian 13 (Trixie) Python image. Swap for an nvidia/cuda image if you need GPU builds.
FROM mcr.microsoft.com/devcontainers/python:3.11-trixie

# System libraries that OpenCV needs at runtime (libGL, glib, etc.).
# Without these, `import cv2` fails with "libGL.so.1: cannot open shared object file".
RUN apt-get update \
	&& apt-get install -y --no-install-recommends \
		libgl1 \
		libglib2.0-0 \
		xauth \
		x11-apps \
		vim-gtk3 \
		ffmpeg \
		libtorch-dev \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

# Keep pip current. Project dependencies are installed via postCreateCommand
# from requirements.txt so they can change without rebuilding the image.
RUN python -m pip install --upgrade pip

# Install PyTorch, OpenCV, and numpy directly into the image.
RUN python -m pip install --no-cache-dir \
	torch \
	torchvision \
	opencv-python \
	numpy \
	anthropic
