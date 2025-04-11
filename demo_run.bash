#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status
TIMESTAMP=66957
VIDEOID=9876543210b
VID="${VIDEOID}_${TIMESTAMP}"

echo "=== Downloading Dataset ==="
gsutil -m cp -R gs://stereo4d/demo .
mv demo stereo4d_dataset
mkdir -p stereo4d_dataset/npz stereo4d_dataset/raw
mv stereo4d_dataset/${VIDEOID}.mp4 stereo4d_dataset/raw 
mv stereo4d_dataset/${VID}.npz stereo4d_dataset/npz

echo "=== Running Rectification ==="
JAX_PLATFORMS=cpu python rectify.py --vid=${VID}

echo "=== Running Optical Flow Inference ==="
python inference_raft.py --vid="$VID"

echo "=== Running Tracking ==="
python tracking.py --vid="$VID"

echo "=== Running Segmentation ==="
python segmentation.py --vid="$VID"

echo "=== Running Track Optimization ==="
python track_optimization.py --vid="$VID"

echo "=== Demo Completed Successfully ==="