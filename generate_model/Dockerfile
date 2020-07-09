FROM nvidia/cuda:9.0-cudnn7-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    python3-dev \
    python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools
RUN pip3 install -U numpy
RUN pip3 install -U pandas
RUN pip3 install -U matplotlib
RUN pip3 install -U networkx
RUN pip3 install -U sklearn
RUN pip3 install -U keras
RUN pip3 install -U tensorflow imutils
RUN pip3 install -U Pillow
RUN pip3 install -U opencv-python

RUN ["mkdir", "results"]

COPY ["covid19_savevers.py", "/"]
COPY ["croppedi2p0", "/croppedi2p0"]

VOLUME /results

ENTRYPOINT python3 covid19_savevers.py
