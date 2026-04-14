#!/usr/bin/env bash

# Usage: run this script from the repository root or any shell with Bash support.
# It launches a batch of YOLOv10 training jobs in Docker and writes outputs under /workspace/output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if command -v id >/dev/null 2>&1; then
	USER_ARGS=(-u "$(id -u):$(id -g)")
else
	USER_ARGS=()
fi

FAILURES=()

record_failure() {
	local label="$1"
	local code="${2:-1}"

	echo "[ERROR] ${label} failed with exit code ${code}, continue to next task" >&2
	FAILURES+=("${label} (exit=${code})")
}

mkdir -p "$PROJECT_ROOT/output"

# Parameter selection:
# - weights: pretrained YOLOv10 checkpoints to evaluate/train, e.g. yolov10n.pt or yolov10m.pt
# - names: experiment names matched to weights, e.g. yolov10n, yolov10m
# - datasets: dataset YAML files to use, e.g. /workspace/data/yolo_sub_split/dataset.yaml
# - dataset_sizes: input image sizes for each dataset, e.g. 640 or 800
# - dataset_tags: short labels used in experiment names, e.g. sub640, sub320, sub800
# - optimizers: optimizer choice for training, e.g. auto
# - batches: batch sizes to try, e.g. 4 or 8
weights=("yolov10n.pt" "yolov10m.pt")
names=("yolov10n" "yolov10m")

datasets=("/workspace/data/yolo_sub_split/dataset.yaml" "/workspace/data/yolo_sub_split_320/dataset.yaml" "/workspace/data/yolo_sub_split_800/dataset.yaml")
dataset_sizes=(640 640 800)
dataset_tags=("sub640" "sub320" "sub800")

optimizers=("auto")
batches=(4 8)

for i in "${!weights[@]}"; do
	for d in "${!datasets[@]}"; do
		for optimizer in "${optimizers[@]}"; do
			for batch in "${batches[@]}"; do
					exp_name="${names[$i]}_${dataset_tags[$d]}_img${dataset_sizes[$d]}_${optimizer}_b${batch}"

				docker run --rm \
					"${USER_ARGS[@]}" \
					--shm-size 8G \
					--gpus=all \
					-v "$PROJECT_ROOT:/workspace" \
					-t waikatodatamining/pytorch-yolov10:2024-06-23_cuda11.7 \
					yolov10_train \
					model=/workspace/models/${weights[$i]} \
					data="${datasets[$d]}" \
					imgsz="${dataset_sizes[$d]}" \
					exist_ok=true \
					project=/workspace/output \
					name="$exp_name" \
					optimizer="$optimizer" \
					patience=20 \
					amp=false \
					batch=$batch \
					epochs=150 \
					mosaic=0 \
					|| record_failure "yolov10:$exp_name" "$?"
			done
		done
		done
done

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
	echo ""
	echo "yolov10 training finished with ${#FAILURES[@]} failure(s):"
	for failure in "${FAILURES[@]}"; do
		echo " - ${failure}"
	done
	exit 1
fi
