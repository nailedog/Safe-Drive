#!/bin/bash
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
TFLITE_SRC="$DIR/face_landmark.tflite"
SCHEMA="$DIR/schema.fbs"
FLATC="/opt/homebrew/bin/flatc"
PYTHON="/opt/homebrew/bin/python3.10"

echo "=== Step 1: Copy model if needed ==="
if [ ! -f "$TFLITE_SRC" ]; then
  cp /tmp/face_landmark.tflite "$TFLITE_SRC"
fi

echo "=== Step 2: Generate JSON ==="
if [ ! -f "$DIR/face_landmark.json" ]; then
  cd "$DIR"
  "$FLATC" -t --strict-json --defaults-json -o . "$SCHEMA" -- face_landmark.tflite
  echo "JSON generated"
else
  echo "JSON already exists"
fi

echo "=== Step 3: Run quantization ==="
"$PYTHON" "$DIR/quantize.py"
