FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install keras tensorflow imutils

RUN ["mkdir", "results"]

COPY ["convert_tf_model_to_tflite.py", "/"]

VOLUME /results
