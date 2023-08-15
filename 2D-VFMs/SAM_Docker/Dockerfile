FROM nvcr.io/nvidia/pytorch:23.06-py3 

WORKDIR /SAM_docker

COPY . /SAM_docker

RUN mkdir -p /SAM_docker/checkpoints \
    && curl 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth' --output /SAM_docker/checkpoints/sam_vit_h_4b8939.pth --compressed

RUN pip install -r requirements.txt

CMD ["python", "run.py"]