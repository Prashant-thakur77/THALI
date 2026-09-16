# Thali — reproducible environment (CPU + Intel iGPU via OpenVINO; CUDA optional for training).
# docker build -t thali . && docker run --rm -it thali make test
FROM python:3.11-slim
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 MUJOCO_GL=egl PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      git ffmpeg libegl1 libgl1 libglib2.0-0 libgomp1 make ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt pyproject.toml ./
RUN pip install uv && uv pip install --system -r requirements.txt
COPY . .
RUN uv pip install --system -e . && python -m souschef_env.build_scene
# EGL works in the container (unlike the Wayland laptop the results were produced on); glfw needs a display.
CMD ["make", "test"]
