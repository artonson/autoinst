FROM vnmsklnk/makeitdense:latest

WORKDIR /workspace

COPY . /workspace

RUN pip install libigl

ENTRYPOINT ["python3", "-u", "run.py"]