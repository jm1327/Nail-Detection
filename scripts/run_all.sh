#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DEFAULT_MODELS=("yolo26" "yolov10" "yolov5" "faster-rcnn")

FAILURES=()

record_failure() {
	local label="$1"
	local code="${2:-1}"

	echo "[ERROR] ${label} failed with exit code ${code}, continue to next task" >&2
	FAILURES+=("${label} (exit=${code})")
}

run_step() {
	local title="$1"
	shift
	local before_failures="${#FAILURES[@]}"
	local after_failures
	local exit_code=0

	echo ""
	echo "========== ${title} =========="
	"$@" || exit_code=$?
	if [[ "$exit_code" -ne 0 ]]; then
		record_failure "$title" "$exit_code"
	fi
	after_failures="${#FAILURES[@]}"
	if [[ "$after_failures" -gt "$before_failures" ]]; then
		echo "========== ${title} finished with failures =========="
	else
		echo "========== ${title} done =========="
	fi
}

train_yolo26() {
	bash "$SCRIPT_DIR/train/yolo26_train.sh"
}

train_yolov10() {
	bash "$SCRIPT_DIR/train/yolov10_train.sh"
}

train_yolov5() {
	bash "$SCRIPT_DIR/train/yolov5_train.sh"
}

train_faster_rcnn() {
	bash "$SCRIPT_DIR/train/faster_rcnn_train.sh"
}

main() {
	local models=("${DEFAULT_MODELS[@]}")
	if [[ "$#" -gt 0 ]]; then
		models=("$@")
	fi

	for model in "${models[@]}"; do
		case "$model" in
			yolo26)
				run_step "Training yolo26" train_yolo26
				;;
			yolov3)
				echo "Unknown model: yolov3"
				echo "Supported: yolo26 yolov10 yolov5 faster-rcnn"
				exit 1
				;;
			yolov10)
				run_step "Training yolov10" train_yolov10
				;;
			yolov5)
				run_step "Training yolov5" train_yolov5
				;;
			faster-rcnn|faster_rcnn)
				run_step "Training faster-rcnn" train_faster_rcnn
				;;
			*)
				echo "Unknown model: $model"
				echo "Supported: yolo26 yolov10 yolov5 faster-rcnn"
				exit 1
				;;
		esac
	done

	if [[ "${#FAILURES[@]}" -gt 0 ]]; then
		local failure
		echo ""
		echo "All requested trainings finished with ${#FAILURES[@]} failure(s):"
		for failure in "${FAILURES[@]}"; do
			echo " - ${failure}"
		done
		exit 1
	fi

	echo ""
	echo "All requested trainings completed."
}

main "$@"