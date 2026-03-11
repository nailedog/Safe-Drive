#!/usr/bin/env python3.10
"""
Patch TF 2.16.2 _get_tensor_details API and run tflite2tensorflow.
"""
import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Monkey-patch before tflite2tensorflow imports TF
import tensorflow as tf

_orig = tf.lite.Interpreter._get_tensor_details

def _patched_get_tensor_details(self, tensor_index, subgraph_index=0):
    return _orig(self, tensor_index, subgraph_index)

tf.lite.Interpreter._get_tensor_details = _patched_get_tensor_details

# Now set up sys.argv and run tflite2tensorflow
DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(DIR)

sys.argv = [
    'tflite2tensorflow',
    '--model_path', 'face_landmark_f32.tflite',
    '--flatc_path', '/opt/homebrew/bin/flatc',
    '--schema_path', os.path.join(DIR, 'schema.fbs'),
    '--model_output_path', os.path.join(DIR, 'savedmodel'),
    '--output_pb',
]

# Load and execute tflite2tensorflow
with open('/opt/homebrew/bin/tflite2tensorflow') as f:
    src = f.read()
# Remove the shebang and execute
src = src[src.find('\n')+1:]
exec(compile(src, 'tflite2tensorflow', 'exec'))
