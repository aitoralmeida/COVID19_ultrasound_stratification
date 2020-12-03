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
RUN pip3 install -U tensorflow==2.3.0 imutils
RUN pip3 install -U Pillow
RUN pip3 install -U opencv-python

RUN ["mkdir", "results"]
RUN ["mkdir", "lus_stratification"]
RUN ["mkdir", "lus_stratification/generate_model"]

COPY ["covid19_configurable.py", "/lus_stratification/generate_model/"]
COPY ["covid19_configurable_temporal.py", "/lus_stratification/generate_model/"]
COPY ["croppedi2p0", "/lus_stratification/generate_model/croppedi2p0"]

VOLUME /results
