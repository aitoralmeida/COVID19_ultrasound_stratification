FROM continuumio/anaconda3

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/* && \
    pip install keras tensorflow imutils

RUN ["mkdir", "results"]

COPY ["convert_keras_to_tflite.py", "/"]
COPY ["convert_keras_to_tflite_optimized.py", "/"]
COPY ["convert_keras_to_tflite_optimized_for_latency.py", "/"]
COPY ["convert_keras_to_tflite_optimized_for_latency_batch_100.py", "/"]

VOLUME /results

ENTRYPOINT python3 convert_keras_to_tflite.py && python3 convert_keras_to_tflite_optimized.py && python3 convert_keras_to_tflite_optimized_for_latency.py && python3 convert_keras_to_tflite_optimized_for_latency_batch_100.py
