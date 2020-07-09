import tensorflow as tf
import keras as k

model = k.models.load_model('/results/covid19_model')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open("/results/covid19_model.tflite", "wb").write(tflite_model)
