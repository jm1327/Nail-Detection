import os
import glob
import csv
import torch
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
from pathlib import Path
import yaml
sys.path.append(os.path.join(ROOT, "yolov5"))
from val import run as val_run

# =========================
# 配置
# =========================
VAL_CONF = 0.38
VAL_IOU = 0.5
DATASET = "val"
DEVICE = 0

DATA_YAML = os.path.join(ROOT, "data", "yolo_sub_split", "dataset.yaml")
MODELS = "yolov5"
MODEL_DIR = f"candidate_model/{MODELS}"
OUTPUT_DIR = DATASET + "_results"
TEMP_DATA_YAML = os.path.join(ROOT, "results", "yolov5_dataset_eval.yaml")

MODEL_PATHS = sorted(glob.glob(os.path.join(MODEL_DIR, "yolo*", "weights", "best.pt")))

def build_eval_dataset_yaml():
    with open(DATA_YAML, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    data_root = Path(ROOT) / "data" / "yolo_sub_split"
    data["train"] = str(data_root / "train" / "images")
    data["val"] = str(data_root / "val" / "images")
    data["test"] = str(data_root / "test" / "images")

    os.makedirs(os.path.dirname(TEMP_DATA_YAML), exist_ok=True)
    with open(TEMP_DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)

    return TEMP_DATA_YAML


def evaluate_model(model_path, data_yaml):
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    print(f"\n===== {model_name} =====")

    # 调用 YOLOv5 val
    results, maps, times = val_run(
        data=data_yaml,
        weights=model_path,
        conf_thres=VAL_CONF,
        iou_thres=VAL_IOU,
        task=DATASET,
        device=DEVICE,
        save_json=True,
        plots=True,
        verbose=True
    )

    # =========================
    # YOLOv5 返回结果解析
    # =========================
    # results: [P, R, mAP50, mAP50-95, ...]
    precision_mean = results[0]
    recall_mean = results[1]
    map50 = results[2]
    map50_95 = results[3]

    print("\n===== METRICS =====")
    print("mAP50:", map50)
    print("mAP50-95:", map50_95)
    print("Precision:", precision_mean)
    print("Recall:", recall_mean)

    # per-class mAP
    map_class0 = maps[0] if len(maps) > 0 else 0.0
    map_class1 = maps[1] if len(maps) > 1 else 0.0

    print("\nPer-class mAP:")
    for i, m in enumerate(maps):
        print(f"Class {i}: {m}")

    return {
        "name": model_name,
        "mAP50": map50,
        "mAP50_95": map50_95,
        "precision_mean": precision_mean,
        "recall_mean": recall_mean,
        "map_class0": map_class0,
        "map_class1": map_class1,
    }


def print_summary_table(rows):
    headers = [
        "name",
        "mAP50",
        "mAP50-95",
        "Precision",
        "Recall",
        "mAP Class0",
        "mAP Class1",
    ]

    table_rows = []
    for row in rows:
        table_rows.append([
            row["name"],
            f"{row['mAP50']:.6f}",
            f"{row['mAP50_95']:.6f}",
            f"{row['precision_mean']:.6f}",
            f"{row['recall_mean']:.6f}",
            f"{row['map_class0']:.6f}",
            f"{row['map_class1']:.6f}",
        ])

    # 打印表格
    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, v in enumerate(r):
            widths[i] = max(widths[i], len(v))

    def fmt_row(vals):
        return " | ".join(v.ljust(widths[i]) for i, v in enumerate(vals))

    print("\n===== SUMMARY TABLE =====")
    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in table_rows:
        print(fmt_row(r))

    # 保存 CSV
    os.makedirs("results", exist_ok=True)
    csv_path = os.path.join("results", f"{MODELS}_eval.csv")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_rows)

    print(f"\nSummary CSV saved to: {csv_path}")


# =========================
# 主函数
# =========================
if __name__ == "__main__":

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch and run on a GPU machine.")
    eval_data_yaml = build_eval_dataset_yaml()

    summary_rows = []

    for m in MODEL_PATHS:
        summary_rows.append(evaluate_model(m, eval_data_yaml))

    if summary_rows:
        print_summary_table(summary_rows)