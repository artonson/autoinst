FROM nvcr.io/nvidia/pytorch:23.06-py3 

ARG ALGORITHM_FOLDER=SAM

ARG CHECKPOINT_URL=https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth

ARG CHECKPOINT_NAME=sam_vit_h_4b8939.pth

WORKDIR /2D-VFMs

COPY ${ALGORITHM_FOLDER} .

RUN mkdir /2D-VFMs/checkpoints \
    && curl ${CHECKPOINT_URL} --output /2D-VFMs/checkpoints/${CHECKPOINT_NAME} --compressed \
    && pip install -r requirements.txt

ENTRYPOINT ["python", "run.py"]
