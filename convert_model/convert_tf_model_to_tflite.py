import tensorflow as tf
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model",
                    type=str,
                    default='mobilenet',
                    nargs="?",
                    help="Model: mobilenet or efficientnet.")
parser.add_argument("--model_version",
                    type=str,
                    default='V1',
                    nargs="?",
                    help="Mobile net version: V1 or V2. Efficient net scaling: B0, B1, B2, B3, B4, B5, B6 or B7.")
args = parser.parse_args()

converter = tf.lite.TFLiteConverter.from_saved_model('/results/covid19_model_' + args.model + args.model_version)
tflite_model = converter.convert()
open('/results/covid19_model_' + args.model + args.model_version + '.tflite', 'wb').write(tflite_model)
