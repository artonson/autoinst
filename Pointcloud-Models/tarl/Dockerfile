FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
##############################################
# You should modify this to match your GPU compute capability
# ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0 8.6 8.9"
##############################################
#jobs for minkowskiengine build
ENV MAX_JOBS=8 
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

ARG DEBIAN_FRONTEND=noninteractive
ARG CHECKPOINT_URL=https://www.ipb.uni-bonn.de/html/projects/tarl/lastepoch199_model_tarl.pt
ARG ALGORITHM_FOLDER=tarl
# Install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    git ninja-build cmake build-essential libopenblas-dev curl \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core  \
    python3-dev \
    python3-pip \ 
    build-essential \
    libopenblas-dev \ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY tarl/requirements.txt .
RUN pip3 install -r requirements.txt

RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine" \
    && cd MinkowskiEngine; python3 setup.py install --force_cuda --blas=openblas


RUN curl ${CHECKPOINT_URL} --output /tmp/tarl.pt 

WORKDIR /tmp

COPY ${ALGORITHM_FOLDER} .
ENTRYPOINT ["python3", "run.py"]