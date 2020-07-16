import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('/results/covid19_model_efficientnet')
tflite_model = converter.convert()
open("/results/covid19_model_efficient_net.tflite", "wb").write(tflite_model)
