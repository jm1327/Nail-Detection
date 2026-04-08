#!/usr/bin/env bash

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

weights=("yolov5n.pt" "yolov5m.pt")
names=("yolov5n" "yolov5m")
optimizers=("SGD")
batches=(8 16)

for i in "${!weights[@]}"; do
	for optimizer in "${optimizers[@]}"; do
		for batch in "${batches[@]}"; do
			exp_name="${names[$i]}_${optimizer}_b${batch}"

			docker run --rm \
				"${USER_ARGS[@]}" \
				--shm-size 8G \
				--gpus=all \
				-v "$PROJECT_ROOT:/workspace" \
				-t waikatodatamining/pytorch-yolov5:2022-11-05_cuda11.1 \
				yolov5_train \
				--img 640 \
				--batch $batch \
				--epochs 150 \
				--data /workspace/data/yolo_sub_split/dataset.yaml \
				--weights /workspace/models/${weights[$i]} \
				--project /workspace/output \
				--name "$exp_name" \
				--optimizer "$optimizer" \
				--patience 20 \
				--exist-ok || record_failure "yolov5:$exp_name" "$?"
		done
	done
done

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
	echo ""
	echo "yolov5 training finished with ${#FAILURES[@]} failure(s):"
	for failure in "${FAILURES[@]}"; do
		echo " - ${failure}"
	done
	exit 1
fi
