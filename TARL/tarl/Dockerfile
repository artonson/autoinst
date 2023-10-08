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
# Install dependencies
RUN apt-get update
RUN apt-get install -y git ninja-build cmake build-essential libopenblas-dev curl \
    xterm xauth openssh-server tmux wget mate-desktop-environment-core  \
    python3-dev \
    python3-pip \ 
    build-essential \
    libopenblas-dev 

RUN pip3 install \
    matplotlib==3.3.4 \
    numpy==1.20.1 \
    tqdm 

RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install pytorch_lightning==2.0.7
RUN git clone --recursive "https://github.com/NVIDIA/MinkowskiEngine"
RUN cd MinkowskiEngine; python3 setup.py install --force_cuda --blas=openblas

ARG ALGORITHM_FOLDER=tarl

WORKDIR /TARL

COPY ${ALGORITHM_FOLDER} .
RUN curl ${CHECKPOINT_URL} --output /TARL/tarl.pt 

ENTRYPOINT ["python3", "run.py"]