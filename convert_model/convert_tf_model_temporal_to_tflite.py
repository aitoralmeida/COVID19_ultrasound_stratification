import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",
                    type=str,
                    default='covid19_model_temporal_mobilenetV1_for_N0_N1',
                    nargs="?",
                    help="Model's folder fullname")
args = parser.parse_args()

model = tf.saved_model.load('/lus_stratification/convert_model/results/' + args.model_name)
concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([None, 5, 224, 224, 3])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
tflite_model = converter.convert()
open('/lus_stratification/convert_model/results/' + args.model_name + '.tflite', 'wb').write(tflite_model)
