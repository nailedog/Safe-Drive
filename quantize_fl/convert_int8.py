#!/usr/bin/env python3.10
"""
Convert SavedModel to INT8 TFLite using TFLiteConverter with representative dataset.
"""
import os, sys
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

DIR       = os.path.dirname(os.path.abspath(__file__))
SAVED_DIR = os.path.join(DIR, 'savedmodel')
CALIB_NPY = os.path.join(DIR, 'calib_faces.npy')
OUT_INT8  = os.path.join(DIR, 'face_landmark_int8.tflite')

print(f"TF {tf.__version__}")

# Load calibration data
if os.path.exists(CALIB_NPY):
    calib_imgs = np.load(CALIB_NPY)
    print(f"Loaded {len(calib_imgs)} calibration images")
else:
    rng = np.random.default_rng(42)
    calib_imgs = rng.normal(0.6, 0.15, (100, 192, 192, 3)).astype(np.float32).clip(0, 1)
    print("Using synthetic calibration data (100 images)")

def representative_dataset():
    for img in calib_imgs:
        yield [img.reshape(1, 192, 192, 3)]

print(f"\nConverting SavedModel at: {SAVED_DIR}")

converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_DIR)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.float32

print("Converting to INT8 (this may take 1-5 minutes)...")
try:
    q_model = converter.convert()
    with open(OUT_INT8, 'wb') as f:
        f.write(q_model)
    print(f"\nSaved: {OUT_INT8} ({len(q_model):,} bytes)")

    # Verify
    iv = tf.lite.Interpreter(model_content=q_model)
    iv.allocate_tensors()
    print("\nVerification:")
    for d in iv.get_input_details():
        print(f"  IN:  {d['shape']} {d['dtype']} quant={d['quantization']}")
    for d in iv.get_output_details():
        print(f"  OUT: {d['shape']} {d['dtype']} quant={d['quantization']}")
    print("\nDone!")

except Exception as e:
    print(f"\nFull INT8 failed: {e}")
    print("\nTrying with TFLITE_BUILTINS (mixed precision)...")

    converter2 = tf.lite.TFLiteConverter.from_saved_model(SAVED_DIR)
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.representative_dataset = representative_dataset
    converter2.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS,
    ]
    converter2.inference_input_type  = tf.uint8
    converter2.inference_output_type = tf.float32

    try:
        q2 = converter2.convert()
        with open(OUT_INT8, 'wb') as f:
            f.write(q2)
        print(f"\nSaved mixed: {OUT_INT8} ({len(q2):,} bytes)")

        iv2 = tf.lite.Interpreter(model_content=q2)
        iv2.allocate_tensors()
        for d in iv2.get_input_details():
            print(f"  IN:  {d['shape']} {d['dtype']}")
        for d in iv2.get_output_details():
            print(f"  OUT: {d['shape']} {d['dtype']}")

    except Exception as e2:
        print(f"Mixed precision also failed: {e2}")
        print("\nTrying dynamic range (weights only)...")

        converter3 = tf.lite.TFLiteConverter.from_saved_model(SAVED_DIR)
        converter3.optimizations = [tf.lite.Optimize.DEFAULT]
        q3 = converter3.convert()
        with open(OUT_INT8, 'wb') as f:
            f.write(q3)
        print(f"\nSaved dynamic-range: {OUT_INT8} ({len(q3):,} bytes)")
        iv3 = tf.lite.Interpreter(model_content=q3)
        iv3.allocate_tensors()
        for d in iv3.get_input_details():
            print(f"  IN:  {d['shape']} {d['dtype']}")
        for d in iv3.get_output_details():
            print(f"  OUT: {d['shape']} {d['dtype']}")
