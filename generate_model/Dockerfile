FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt-get update && apt-get upgrade -y --allow-unauthenticated && \
    apt-get install -y --no-install-recommends --allow-unauthenticated \
    build-essential \
    git \
    python3-dev \
    python3-pip

RUN pip3 install --upgrade pip
RUN pip3 install -U setuptools
RUN pip3 install -U numpy==1.16.0
RUN pip3 install -U pandas
RUN pip3 install -U matplotlib
RUN pip3 install -U networkx
RUN pip3 install -U sklearn
RUN pip3 install -U keras
RUN pip3 install -U tf-nightly imutils
RUN pip3 install -U Pillow
RUN pip3 install -U opencv-python

RUN ["mkdir", "results"]

COPY ["covid19_configurable.py", "/"]
COPY ["croppedi2p0", "/croppedi2p0"]

VOLUME /results
