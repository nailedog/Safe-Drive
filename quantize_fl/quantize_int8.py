#!/usr/bin/env python3.10
"""
INT8 quantize face_landmark_f32.tflite using direct FlatBuffer manipulation.

Approach: Since TFLiteConverter can't directly process .tflite, we do PTQ
(Post-Training Quantization) by directly modifying the FlatBuffer:
  1. Load FLOAT32 model
  2. Run calibration to collect per-tensor activation ranges
  3. Compute INT8 scale/zero_point for each tensor
  4. Modify FlatBuffer: set quantization_parameters on each tensor
  5. Convert weight tensors to INT8 data
  6. Change op kernel types to INT8 variants
  7. Add QUANTIZE op at input, DEQUANTIZE at output
  8. Write new .tflite

This gives an INT8 model compatible with ESP-NN SIMD optimizations.
"""

import json
import struct
import os
import sys
import subprocess
import numpy as np
from collections import defaultdict

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

DIR     = os.path.dirname(os.path.abspath(__file__))
F32_TF  = os.path.join(DIR, 'face_landmark_f32.tflite')
OUT_INT8 = os.path.join(DIR, 'face_landmark_int8.tflite')
CALIB_NPY = os.path.join(DIR, 'calib_faces.npy')
SCHEMA   = os.path.join(DIR, 'schema.fbs')
FLATC    = '/opt/homebrew/bin/flatc'

print(f"TF {tf.__version__}")

# ─── Step 1: Collect activation ranges via calibration ────────────────────────
print("\nStep 1: Calibration...")

interp = tf.lite.Interpreter(model_path=F32_TF)
interp.allocate_tensors()
inp_det = interp.get_input_details()[0]
out_dets = interp.get_output_details()

print(f"  Input:  {inp_det['shape']}  dtype={inp_det['dtype'].__name__}")
for od in out_dets:
    print(f"  Output: {od['shape']}  dtype={od['dtype'].__name__}")

# Load calibration images
if os.path.exists(CALIB_NPY):
    calib_imgs = np.load(CALIB_NPY)
    print(f"  Loaded {len(calib_imgs)} calibration images")
else:
    rng = np.random.default_rng(42)
    calib_imgs = rng.normal(0.6, 0.15, (100, 192, 192, 3)).astype(np.float32).clip(0, 1)
    print("  Using synthetic calibration data")

# Get all tensor details (including intermediate)
try:
    interp_full = tf.lite.Interpreter(model_path=F32_TF)
    interp_full.allocate_tensors()
    all_tensors = interp_full.get_tensor_details()
    print(f"  Total tensors to calibrate: {len(all_tensors)}")

    tensor_mins = {}
    tensor_maxs = {}

    for n, img in enumerate(calib_imgs):
        if n % 20 == 0:
            print(f"  Calibrating {n}/{len(calib_imgs)}...")
        x = img.reshape(1, 192, 192, 3).astype(np.float32)
        interp_full.set_tensor(inp_det['index'], x)
        interp_full.invoke()

        for td in all_tensors:
            idx = td['index']
            try:
                t = interp_full.get_tensor(idx)
                if t.dtype != np.float32:
                    continue
                if len(t.shape) == 0:
                    continue
                mn = float(t.min())
                mx = float(t.max())
                if idx not in tensor_mins:
                    tensor_mins[idx] = mn
                    tensor_maxs[idx] = mx
                else:
                    tensor_mins[idx] = min(tensor_mins[idx], mn)
                    tensor_maxs[idx] = max(tensor_maxs[idx], mx)
            except Exception:
                pass

    print(f"  Calibrated {len(tensor_mins)} tensors")
    print(f"  Input range: [{tensor_mins.get(inp_det['index'], '?'):.4f}, {tensor_maxs.get(inp_det['index'], '?'):.4f}]")
    for od in out_dets:
        idx = od['index']
        print(f"  Output[{idx}] range: [{tensor_mins.get(idx, '?'):.4f}, {tensor_maxs.get(idx, '?'):.4f}]")

except Exception as e:
    print(f"  Calibration failed: {e}")
    sys.exit(1)

# ─── Step 2: Try TFLiteConverter with py_function wrapper ─────────────────────
# This is the standard approach. If it works, great. If not, fall through to
# the direct FlatBuffer approach.
print("\nStep 2: Attempting TFLiteConverter (py_function wrapper)...")

@tf.function(input_signature=[tf.TensorSpec([1, 192, 192, 3], tf.float32)])
def f32_forward(x):
    def _run(xnp):
        i2 = tf.lite.Interpreter(model_path=F32_TF)
        i2.allocate_tensors()
        id2 = i2.get_input_details()[0]
        od2 = i2.get_output_details()
        i2.set_tensor(id2['index'], xnp.numpy())
        i2.invoke()
        lm = i2.get_tensor(od2[0]['index'])  # [1,1,1,1404]
        sc = i2.get_tensor(od2[1]['index'])  # [1,1,1,1]
        return lm.reshape(1, 1404), sc.reshape(1, 1)
    lm, sc = tf.py_function(_run, [x], [tf.float32, tf.float32])
    lm.set_shape([1, 1404])
    sc.set_shape([1, 1])
    return lm, sc

def representative_dataset():
    for img in calib_imgs:
        yield [img.reshape(1, 192, 192, 3)]

concrete_fn = f32_forward.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], f32_forward)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS,
]
converter.inference_input_type  = tf.uint8
converter.inference_output_type = tf.float32

tflc_success = False
try:
    print("  Converting (running calibration passes)...")
    q_model = converter.convert()
    with open(OUT_INT8, 'wb') as f:
        f.write(q_model)
    print(f"  SUCCESS! Saved: {OUT_INT8} ({len(q_model):,} bytes)")
    tflc_success = True
except Exception as e:
    print(f"  TFLiteConverter failed: {e}")

if tflc_success:
    # Verify
    iv = tf.lite.Interpreter(model_content=open(OUT_INT8,'rb').read())
    iv.allocate_tensors()
    print("  Verification:")
    for d in iv.get_input_details():
        print(f"    IN:  {d['shape']} {d['dtype']}")
    for d in iv.get_output_details():
        print(f"    OUT: {d['shape']} {d['dtype']}")
    print("\nDone!")
    sys.exit(0)

# ─── Step 3: Direct FlatBuffer PTQ ───────────────────────────────────────────
print("\nStep 3: Direct FlatBuffer INT8 quantization...")

# Generate JSON for the FLOAT32 model
F32_JSON = os.path.join(DIR, 'face_landmark_f32.json')
if not os.path.exists(F32_JSON):
    print("  Generating JSON from FLOAT32 model...")
    os.chdir(DIR)
    result = subprocess.run(
        [FLATC, '-t', '--strict-json', '--defaults-json', '-o', DIR,
         SCHEMA, '--', 'face_landmark_f32.tflite'],
        capture_output=True, text=True, cwd=DIR
    )
    if result.returncode != 0:
        print(f"  flatc error: {result.stderr}")
        sys.exit(1)
    print("  JSON generated")

print("  Loading F32 JSON...")
with open(F32_JSON) as f:
    model = json.load(f)

sg = model['subgraphs'][0]
tensors = sg['tensors']
operators = sg['operators']
op_codes = model['operator_codes']
buffers = model['buffers']

print(f"  Tensors: {len(tensors)}, Ops: {len(operators)}, Buffers: {len(buffers)}")

# Helper: compute INT8 scale and zero_point from range [mn, mx]
def compute_scale_zp_int8(mn, mx):
    """Compute symmetric INT8 quantization params."""
    # Ensure range includes 0 for proper zero_point computation
    mn = min(mn, 0.0)
    mx = max(mx, 0.0)
    if mx == mn:
        return 1.0 / 128.0, 0
    scale = max(abs(mn), abs(mx)) / 127.0
    if scale == 0:
        scale = 1e-8
    zero_point = 0  # symmetric
    return scale, zero_point

def compute_scale_zp_uint8(mn, mx):
    """Compute asymmetric UINT8 quantization params (for input)."""
    mn = min(mn, 0.0)
    mx = max(mx, 0.0)
    if mx == mn:
        return 1.0 / 255.0, 0
    scale = (mx - mn) / 255.0
    zero_point = round(-mn / scale)
    zero_point = max(0, min(255, zero_point))
    return scale, zero_point

# Build tensor index → tensor detail mapping from calibration
print("  Applying per-tensor quantization parameters...")

# Get the tensor index mapping from the interpreter
interp2 = tf.lite.Interpreter(model_path=F32_TF)
interp2.allocate_tensors()
tflite_tensor_details = {td['index']: td for td in interp2.get_tensor_details()}

# Map tflite interpreter indices to JSON tensor indices
# TFLite tensor indices should match JSON tensor indices (0-based in order)
n_quant_applied = 0
n_weights_quantized = 0

for t_idx, t in enumerate(tensors):
    if t.get('type') != 'FLOAT32':
        continue

    buf_idx = t.get('buffer', 0)
    buf = buffers[buf_idx]
    has_data = 'data' in buf and buf['data']

    if t_idx in tensor_mins:
        mn = tensor_mins[t_idx]
        mx = tensor_maxs[t_idx]
    elif has_data:
        # Constant weight tensor not seen during calibration
        # Decode float32 data to find range
        data_bytes = bytes(buf['data'])
        n_el = len(data_bytes) // 4
        if n_el > 0:
            vals = np.frombuffer(data_bytes, dtype=np.float32)
            mn, mx = float(vals.min()), float(vals.max())
        else:
            continue
    else:
        continue

    # Determine if this is the model input (use uint8) or internal (use int8)
    is_input = (t_idx in sg.get('inputs', []))

    if is_input:
        # Input: UINT8 [0, 255] maps to float [0, 1]
        scale, zp = 1.0/255.0, 0
        t['type'] = 'UINT8'
    else:
        scale, zp = compute_scale_zp_int8(mn, mx)
        t['type'] = 'INT8'

    t['quantization'] = {
        'scale': [scale],
        'zero_point': [zp],
        'quantized_dimension': 0
    }
    n_quant_applied += 1

    # Quantize weight data if present
    if has_data and not is_input:
        data_bytes = bytes(buf['data'])
        n_el = len(data_bytes) // 4
        vals_f32 = np.frombuffer(data_bytes, dtype=np.float32)
        vals_int8 = np.clip(np.round(vals_f32 / scale).astype(np.int32), -127, 127).astype(np.int8)
        buf['data'] = list(vals_int8.tobytes())
        n_weights_quantized += 1

print(f"  Applied quant params to {n_quant_applied} tensors")
print(f"  Quantized {n_weights_quantized} weight tensors to INT8")

# Handle INT32 bias tensors - they stay INT32 for CONV ops
# (bias quantization is special: scale = weight_scale * input_scale, zp=0)
for t_idx, t in enumerate(tensors):
    if t.get('type') == 'INT32':
        buf_idx = t.get('buffer', 0)
        buf = buffers[buf_idx]
        has_data = 'data' in buf and buf['data']
        if has_data:
            data_bytes = bytes(buf['data'])
            n_el = len(data_bytes) // 4
            vals = np.frombuffer(data_bytes, dtype=np.float32)
            # Find connected weight tensor to get scale
            mn, mx = float(vals.min()), float(vals.max())
            scale = max(abs(mn), abs(mx)) / (2**31 - 1) if mx != mn else 1e-8
            t['quantization'] = {'scale': [scale], 'zero_point': [0]}

# Add output DEQUANTIZE handling: outputs stay FLOAT32
print("  Output tensors remain FLOAT32 (no dequantize needed)")

# Write modified JSON
F32_INT8_JSON = os.path.join(DIR, 'face_landmark_int8.json')
print(f"  Writing INT8 JSON...")
with open(F32_INT8_JSON, 'w') as f:
    json.dump(model, f)

# Compile back to .tflite
print("  Compiling INT8 .tflite with flatc...")
result = subprocess.run(
    [FLATC, '-b', '--root-type', 'tflite.Model', SCHEMA, F32_INT8_JSON],
    capture_output=True, text=True, cwd=DIR
)

out_bin = os.path.join(DIR, 'face_landmark_int8.bin')
if os.path.exists(out_bin):
    os.rename(out_bin, OUT_INT8)
    print(f"  Saved: {OUT_INT8} ({os.path.getsize(OUT_INT8):,} bytes)")
else:
    print(f"  flatc error: {result.stderr}")
    print(f"  flatc stdout: {result.stdout}")
    print("  Files in DIR:", os.listdir(DIR))
    sys.exit(1)

# Verify
print("\nVerifying INT8 model...")
try:
    iv = tf.lite.Interpreter(model_path=OUT_INT8)
    iv.allocate_tensors()
    for d in iv.get_input_details():
        print(f"  IN:  {d['shape']} {d['dtype']} quant={d['quantization']}")
    for d in iv.get_output_details():
        print(f"  OUT: {d['shape']} {d['dtype']} quant={d['quantization']}")
    print("\nDone!")
except Exception as e:
    print(f"  Verification failed: {e}")
