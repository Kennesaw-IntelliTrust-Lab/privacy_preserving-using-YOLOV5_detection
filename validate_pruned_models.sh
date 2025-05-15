#!/bin/bash

WEIGHTS_DIR="/mnt/bst/hxu10/hxu10/chanti/yolov5/runs/train/exp/weights"
DATA_YAML="/mnt/bst/hxu10/hxu10/chanti/dataset/data.yaml"

for ratio in 2 5 10 20 30 40 50
do
    echo "=== Validating direct_pruned_${ratio}.pt ==="
    python val.py \
      --weights "${WEIGHTS_DIR}/direct_pruned_${ratio}.pt" \
      --data "$DATA_YAML" \
      --task test \
      --img 640 \
      --project runs/val \
      --name direct_pruned_${ratio}_val \
      --exist-ok
done
