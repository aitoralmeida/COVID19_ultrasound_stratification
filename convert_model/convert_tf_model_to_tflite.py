import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name",
                    type=str,
                    default='covid19_model_mobilenetV1_for_N0_N1',
                    nargs="?",
                    help="Model's folder fullname")
args = parser.parse_args()

converter = tf.lite.TFLiteConverter.from_saved_model('/results/' + args.model_name)
tflite_model = converter.convert()
open('/results/' + args.model_name + '.tflite', 'wb').write(tflite_model)
