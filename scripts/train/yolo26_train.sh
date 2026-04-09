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

mkdir -p "$PROJECT_ROOT/cache" "$PROJECT_ROOT/config" "$PROJECT_ROOT/output"

weights=("yolo26l.pt" "yolo26m.pt")
names=("yolo26l" "yolo26m")
datasets=("/workspace/data/yolo_sub_split/dataset.yaml")
dataset_sizes=(640)
dataset_tags=("sub640")
optimizers=("auto")
batches=(8)

exp_id=0

for i in "${!weights[@]}"; do
	for d in "${!datasets[@]}"; do
		for optimizer in "${optimizers[@]}"; do
			for batch in "${batches[@]}"; do
				exp_id=$((exp_id + 1))
				exp_name="${names[$i]}_${dataset_tags[$d]}_img${dataset_sizes[$d]}_${optimizer}_b${batch}"

				echo "Running experiment $exp_id: $exp_name"

				docker run --rm \
					"${USER_ARGS[@]}" \
					--shm-size 8G \
					--gpus=all \
					-v "$PROJECT_ROOT:/workspace" \
					-v "$PROJECT_ROOT/cache:/cache" \
					-v "$PROJECT_ROOT/config:/config" \
					-t waikatodatamining/pytorch-yolo26:8.4.16_cuda12.6 \
					yolo26_train \
					model=/workspace/models/${weights[$i]} \
					data=${datasets[$d]} \
					imgsz=${dataset_sizes[$d]} \
					exist_ok=true \
					project=/workspace/output/ \
					name="$exp_name" \
					optimizer="$optimizer" \
					patience=20 \
					amp=false \
					batch=$batch \
					epochs=150 \
					mosaic=0 \
					conf=0.001 \
					save_txt=True \
					save_conf=True \
					save_json=True \
					|| record_failure "yolo26:$exp_name" "$?"
			done
		done
		done
	done

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
	echo ""
	echo "yolo26 training finished with ${#FAILURES[@]} failure(s):"
	for failure in "${FAILURES[@]}"; do
		echo " - ${failure}"
	done
	exit 1
fi
