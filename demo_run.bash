#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

echo "=== Downloading Dataset ==="
gsutil -m cp -R gs://stereo4d/demo .
mv demo stereo4d_dataset
mkdir -p stereo4d_dataset/npz stereo4d_dataset/raw
mv stereo4d_dataset/0123456789a.mp4 stereo4d_dataset/raw 
mv stereo4d_dataset/0123456789a_6762638.npz stereo4d_dataset/npz

echo "=== Running Rectification ==="
JAX_PLATFORMS=cpu python rectify.py --vid=0123456789a_6762638

echo "=== Running Optical Flow Inference ==="
python inference_raft.py --vid=0123456789a_6762638

echo "=== Running Tracking ==="
python tracking.py --vid=0123456789a_6762638

echo "=== Running Segmentation ==="
python segmentation.py --vid=0123456789a_6762638

echo "=== Running Track Optimization ==="
python track_optimization.py --vid=0123456789a_6762638

echo "=== Demo Completed Successfully ==="