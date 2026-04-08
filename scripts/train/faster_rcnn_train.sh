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

mkdir -p "$PROJECT_ROOT/cache" "$PROJECT_ROOT/output"

base_lrs=(0.001 0.002 0.003 0.005 0.01)
train_batch_sizes=(1 2)

for base_lr in "${base_lrs[@]}"; do
	for train_bs in "${train_batch_sizes[@]}"; do
		lr_tag="${base_lr//./p}"
		export_config_dir="/workspace/output/faster_rcnn_r50_fpn_1x_coco_lr${lr_tag}_bs${train_bs}"
		export_config_file="${export_config_dir}/faster_rcnn_r50_fpn_1x_coco.yml"
		save_dir="/workspace/output/faster_rcnn_r50_fpn_1x_coco_lr${lr_tag}_bs${train_bs}"
		output_eval="${save_dir}/eval"
		final_weights="${save_dir}/model_final"

		echo "Exporting faster-rcnn config for base_lr=${base_lr}, TrainReader.batch_size=${train_bs}"
		docker run --rm \
			"${USER_ARGS[@]}" \
			--gpus=all \
			-v "$PROJECT_ROOT:/workspace" \
			-v "$PROJECT_ROOT/cache:/.cache" \
			-v "$PROJECT_ROOT/cache:/opt/PaddleDetection/~/.cache" \
			-t waikatodatamining/paddledetection:2.8.0_cuda11.8 \
			paddledet_export_config \
			-i /opt/PaddleDetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.yml \
			-o "$export_config_file" \
			-O "$export_config_dir" \
			-t /workspace/data/coco_sub_split/train/annotations.json \
			-v /workspace/data/coco_sub_split/val/annotations.json \
			--save_interval 999 \
			--num_epochs 150 \
			--num_classes 2 \
			-a \
			"pretrain_weights:/workspace/models/faster_rcnn_r50_fpn_1x_coco.pdparams" \
			"EvalReader.sample_transforms.1.Resize.target_size:800,800" \
			"LearningRate.base_lr:${base_lr}" \
			"RPNHead.anchor_generator.anchor_sizes:[[8],[16],[32],[64],[128]]" \
			"TestReader.sample_transforms.1.Resize.target_size:800,800" \
			"TrainReader.sample_transforms.1.RandomResize.target_size:800,800" \
			"TrainReader.batch_size:${train_bs}" \
			"save_dir:${save_dir}" \
			"output_eval:${output_eval}" \
			"weights:${final_weights}" \
			"snapshot_epoch:50" \
			|| {
				record_failure "faster-rcnn:export-config:lr=${base_lr}:bs=${train_bs}" "$?"
				continue
			}

		train_config="$export_config_file"
		log_file="$PROJECT_ROOT/output/fasterrcnn_lr${lr_tag}_bs${train_bs}.log"

		echo "Running faster-rcnn config: ${train_config}"
		docker run --rm \
			"${USER_ARGS[@]}" \
			--shm-size 8G \
			--gpus=all \
			-v "$PROJECT_ROOT:/workspace" \
			-v "$PROJECT_ROOT/cache:/.cache" \
			-v "$PROJECT_ROOT/cache:/opt/PaddleDetection/~/.cache" \
			-t waikatodatamining/paddledetection:2.8.0_cuda11.8 \
			paddledet_train \
			-c "${train_config}" \
			--eval \
			-o use_gpu=true 2>&1 | tee "$log_file" || record_failure "faster-rcnn:${train_config}" "$?"
		done
	done

if [[ "${#FAILURES[@]}" -gt 0 ]]; then
	echo ""
	echo "faster-rcnn training finished with ${#FAILURES[@]} failure(s):"
	for failure in "${FAILURES[@]}"; do
		echo " - ${failure}"
	done
	exit 1
fi
