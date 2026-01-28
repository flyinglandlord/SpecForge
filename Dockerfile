# ===============================
# 1. CUDA 12.8 官方运行时
# ===============================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu20.04

# ===============================
# 2. 基本环境变量
# ===============================
ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# cache（集群必备）
ENV HF_HOME=/workspace/.cache/huggingface
ENV TORCH_HOME=/workspace/.cache/torch

# ===============================
# 3. 系统依赖
# ===============================
RUN apt-get update && apt-get install -y \
    wget \
    git \
    ca-certificates \
    bash \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# 4. 安装 Miniconda
# ===============================
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash Miniconda3-latest-Linux-x86_64.sh -b -p $CONDA_DIR \
    && rm Miniconda3-latest-Linux-x86_64.sh \
    && conda clean -afy

RUN conda tos accept --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --channel https://repo.anaconda.com/pkgs/r

SHELL ["bash", "-c"]

# ===============================
# 5. 创建两个 Conda env（Python 3.11）
# ===============================
RUN conda create -y -n train_env python=3.11 \
 && conda create -y -n eval_env python=3.11 \
 && conda clean -afy

# ===============================
# 6. pip 依赖（分别）
# ===============================
COPY benchmarks/EAGLE/requirements.txt /tmp/eval_env.txt

# ⚠️ PyTorch cu128：推荐用 nightly / 官方 index
RUN conda run -n eval_env pip install --upgrade pip \
 && conda run -n eval_env pip install \
      --index-url https://download.pytorch.org/whl/nightly/cu128 \
      torch torchvision torchaudio \
 && conda run -n eval_env pip install -r /tmp/eval_env.txt
# ===============================
# 7. 工程代码
# ===============================
WORKDIR /workspace
COPY . /workspace

RUN conda run -n train_env pip install --upgrade pip \
 && conda run -n train_env pip install \
      --index-url https://download.pytorch.org/whl/nightly/cu128 \
      torch torchvision torchaudio \
 && conda run -n train_env pip install -e .

# ===============================
# 8. 默认 shell
# ===============================
CMD ["/bin/bash"]