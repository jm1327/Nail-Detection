# summarize_results.py

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


MAP_COLUMN_CANDIDATES = [
    "metrics/mAP50-95(B)",
    "metrics/mAP50-95",
    "metrics/mAP50(B)",
    "metrics/mAP50",
    "metrics/mAP_0.5:0.95",
    "metrics/mAP_0.5",
]

PRECISION_COLUMN_CANDIDATES = [
    "metrics/precision(B)",
    "metrics/precision",
]

RECALL_COLUMN_CANDIDATES = [
    "metrics/recall(B)",
    "metrics/recall",
]

GT_JSON = "data/coco_sub_split/val/annotations.json"

FRCNN_EVAL_FILENAME_CANDIDATES = [
    "eval_val/bbox.json",
]

COCO_METRIC_NAMES = [
    "AP@[0.50:0.95]|all|100",
    "AP@[0.50]|all|100",
    "AP@[0.75]|all|100",
    "AP@[0.50:0.95]|small|100",
    "AP@[0.50:0.95]|medium|100",
    "AP@[0.50:0.95]|large|100",
    "AR@[0.50:0.95]|all|1",
    "AR@[0.50:0.95]|all|10",
    "AR@[0.50:0.95]|all|100",
    "AR@[0.50:0.95]|small|100",
    "AR@[0.50:0.95]|medium|100",
    "AR@[0.50:0.95]|large|100",
]


def get_map_column(columns):
    for candidate in MAP_COLUMN_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def get_first_available_column(columns, candidates):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def find_iou_index(iou_thrs: np.ndarray, target: float) -> int:
    idxs = np.where(np.isclose(iou_thrs, target))[0]
    if idxs.size == 0:
        return -1
    return int(idxs[0])


def mean_valid(values: np.ndarray) -> float:
    valid = values[values > -1]
    if valid.size == 0:
        return 0.0
    return float(np.mean(valid))


def per_class_metrics(coco_eval: Any, gt: Any) -> list[dict]:
    """Extract per-class AP/Precision/Recall from COCOeval tensors.

    precision shape: [T, R, K, A, M]
    recall shape:    [T, K, A, M]
    """

    precision = coco_eval.eval["precision"]
    recall = coco_eval.eval["recall"]
    iou_thrs = np.array(coco_eval.params.iouThrs)
    cat_ids = list(coco_eval.params.catIds)

    area_idx = 0  # area=all
    max_det_idx = len(coco_eval.params.maxDets) - 1  # usually maxDets=100
    iou50_idx = find_iou_index(iou_thrs, 0.5)

    rows: list[dict] = []
    for k, cat_id in enumerate(cat_ids):
        cat_name = str(gt.cats.get(cat_id, {}).get("name", cat_id))

        ap_50_95 = mean_valid(precision[:, :, k, area_idx, max_det_idx])
        ap_50 = mean_valid(precision[iou50_idx, :, k, area_idx, max_det_idx]) if iou50_idx >= 0 else 0.0

        p_50 = ap_50
        r_50 = mean_valid(np.array([recall[iou50_idx, k, area_idx, max_det_idx]])) if iou50_idx >= 0 else 0.0

        rows.append(
            {
                "category_id": int(cat_id),
                "category_name": cat_name,
                "AP@[0.50:0.95]|all|100": ap_50_95,
                "AP@[0.50]|all|100": ap_50,
                "AR@[0.50]|all|100": r_50,
            }
        )

    return rows


def find_faster_rcnn_prediction_file(model_dir: Path) -> Path | None:
    pred_path = model_dir / FRCNN_EVAL_FILENAME_CANDIDATES[0]
    if pred_path.exists():
        return pred_path

    return None


def evaluate_faster_rcnn(gt_json=GT_JSON, output_dir="output"):
    gt_path = Path(gt_json)

    if not gt_path.exists():
        raise SystemExit(f"GT file not found: {gt_path}")

    gt = COCO(str(gt_path))
    output_path = Path(output_dir)

    if not output_path.exists():
        raise SystemExit(f"Output directory not found: {output_path}")

    results = []

    for model_dir in sorted(output_path.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith("faster_rcnn"):
            continue

        pred_path = find_faster_rcnn_prediction_file(model_dir)
        if pred_path is None:
            print(f"[WARN] Skip {model_dir}: no bbox.json found")
            continue

        print(f"Evaluating Faster R-CNN: {model_dir.name}")
        dt = gt.loadRes(str(pred_path))

        coco_eval = COCOeval(gt, dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        metrics = {}
        for idx, name in enumerate(COCO_METRIC_NAMES):
            metrics[name] = float(coco_eval.stats[idx])

        class_metrics = per_class_metrics(coco_eval, gt)

        payload = {
            "gt_json": str(gt_path),
            "pred_json": str(pred_path),
            "metrics": metrics,
            "per_class_metrics": class_metrics,
        }

        out_path = model_dir / "coco_eval_result.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"Saved: {out_path}")
        results.append(payload)

    return results


def summarize_faster_rcnn_results(results: list[dict]) -> pd.DataFrame:
    rows = []

    for payload in results:
        pred_path = Path(payload["pred_json"])
        model_name = pred_path.parent.parent.name if pred_path.parent.name == "eval_val" else pred_path.parent.name

        rows.append(
            {
                "model": model_name,
                "epoch": None,
                "best mAP50-95": payload["metrics"]["AP@[0.50:0.95]|all|100"],
                "mAP50": payload["metrics"]["AP@[0.50]|all|100"],
                "precision": None,
                "recall": None,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["model", "epoch", "best mAP50-95", "mAP50", "precision", "recall"])

    return pd.DataFrame(rows)


def summarize(output_dir="output"):
    results = []

    for model_name in os.listdir(output_dir):
        if not model_name.startswith("yolo"):
            continue

        model_path = os.path.join(output_dir, model_name)
        if not os.path.isdir(model_path):
            continue

        csv_path = os.path.join(model_path, "results.csv")

        if not os.path.exists(csv_path):
            continue

        df = pd.read_csv(csv_path, skipinitialspace=True)
        df.columns = df.columns.str.strip()

        map50_95_col = get_map_column(df.columns)
        if map50_95_col is None:
            print(f"[WARN] Skip {csv_path}: no supported mAP50-95 column found")
            continue

        map50_col = get_first_available_column(df.columns, ["metrics/mAP50(B)", "metrics/mAP50", "metrics/mAP_0.5"])
        precision_col = get_first_available_column(df.columns, PRECISION_COLUMN_CANDIDATES)
        recall_col = get_first_available_column(df.columns, RECALL_COLUMN_CANDIDATES)

        best_row = df.loc[df[map50_95_col].idxmax()]

        results.append({
            "model": model_name,
            "epoch": best_row.get("epoch"),
            "best mAP50-95": best_row[map50_95_col],
            "mAP50": best_row.get(map50_col) if map50_col is not None else None,
            "precision": best_row.get(precision_col) if precision_col is not None else None,
            "recall": best_row.get(recall_col) if recall_col is not None else None,
        })

    if not results:
        return pd.DataFrame(columns=["model", "epoch", "best mAP50-95", "mAP50", "precision", "recall"])

    return pd.DataFrame(results).sort_values(by="best mAP50-95", ascending=False)


if __name__ == "__main__":
    faster_rcnn_results = evaluate_faster_rcnn()
    yolo_df = summarize()
    faster_rcnn_df = summarize_faster_rcnn_results(faster_rcnn_results)
    df = pd.concat([yolo_df, faster_rcnn_df], ignore_index=True)
    if not df.empty:
        df = df.sort_values(by="best mAP50-95", ascending=False)
    print(df)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "model_summary.csv", index=False)