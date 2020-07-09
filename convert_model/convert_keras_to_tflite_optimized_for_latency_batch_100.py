import tensorflow as tf
import keras as k

model = k.models.load_model('/results/covid19_model')

batch_size = 100
input_shape = model.inputs[0].shape.as_list()
input_shape[0] = batch_size
func = tf.function(model).get_concrete_function(
    tf.TensorSpec(input_shape, model.inputs[0].dtype))

converter = tf.lite.TFLiteConverter.from_concrete_functions([func])
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_LATENCY]
tflite_model = converter.convert()
open("/results/covid19_model_optimized_for_latency_batch_100.tflite", "wb").write(tflite_model)
