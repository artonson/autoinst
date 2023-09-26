FROM nvcr.io/nvidia/pytorch:23.06-py3 

ARG ALGORITHM_FOLDER=SLIC

WORKDIR /2D-VFMs

COPY ${ALGORITHM_FOLDER} .

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "run.py"]