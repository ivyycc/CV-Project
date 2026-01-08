#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Configurable paths
# -----------------------------
IMG_DIR="./data/images"
MASK_DIR="./data/masks"

UNET_WEIGHTS="./models/unet_best.pth"         

GRAPH_OUT="./graph.json"
FINAL_OUT="./final.png"

FEATURES_PATH="./outputs/features_emb.pkl"

# -----------------------------
# 0. Sanity checks
# -----------------------------
if [ ! -d "$IMG_DIR" ]; then
  echo "ERROR: Image directory '$IMG_DIR' not found."
  exit 1
fi

if [ ! -d "$MASK_DIR" ]; then
  echo "Creating mask directory at '$MASK_DIR'..."
  mkdir -p "$MASK_DIR"
fi

if [ ! -f "$UNET_WEIGHTS" ]; then
  echo "ERROR: U-Net weights not found at '$UNET_WEIGHTS'."
  echo "Please place your trained segmentation model there."
  exit 1
fi


if [ ! -f "$FEATURES_PATH" ]; then
  echo "ERROR: Features file '$FEATURES_PATH' not found."
  echo "Make sure outputs/features_emb.pkl is included ."
  exit 1
fi


# -----------------------------
# 1. Generate segmentation masks for UNLABELLED images
#    (Should skip the 500 already-labelled masks in ./data/masks/)
# -----------------------------
echo "=== Step 1: Predicting masks for unlabeled images ==="

python inference.py \
  --images_dir "$IMG_DIR" \
  --masks_dir "$MASK_DIR" \
  --checkpoint "$UNET_WEIGHTS" 

# Notes:
# - predict_unlabelled.py should:
#   * iterate over all images in ./data/images/
#   * check whether a mask already exists in ./data/masks/ for that image
#   * only run inference for images WITHOUT an existing mask
#   * save new masks into ./data/pred_masks/ with the correct naming convention

# -----------------------------
# 2. Build final adjacency graph (graph.json)
# -----------------------------
echo "=== Step 2: Building adjacency graph ==="


python build_graph.py \
  --features "$FEATURES_PATH" \
  --out "./graph.json"

echo "=== Done! ==="
echo "Adjacency graph: $GRAPH_OUT"
