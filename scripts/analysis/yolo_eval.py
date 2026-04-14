import os
import glob
import csv
from ultralytics import YOLO

# =========================
# 配置
# =========================
VAL_CONF = 0.33
VAL_IOU = 0.5
DATASET = "val"

DATA_YAML = "data/yolo_sub_split_320/dataset.yaml"  
MODELS = "yolo_img320"
MODEL_DIR = f"candidate_model/{MODELS}"
OUTPUT_DIR = DATASET + "_results"

MODEL_PATHS = sorted(glob.glob(os.path.join(MODEL_DIR, "yolo*", "weights", "best.pt")))


def _safe_float(x, default=0.0):
    if x is None:
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _get_class_pr(metrics, class_idx):
    p_list = getattr(metrics.box, "p", None)
    r_list = getattr(metrics.box, "r", None)

    if p_list is not None and r_list is not None and len(p_list) > class_idx and len(r_list) > class_idx:
        return _safe_float(p_list[class_idx]), _safe_float(r_list[class_idx])

    if hasattr(metrics.box, "class_result"):
        try:
            p_i, r_i, _, _ = metrics.box.class_result(class_idx)
            return _safe_float(p_i), _safe_float(r_i)
        except Exception:
            return 0.0, 0.0

    return 0.0, 0.0


def print_summary_table(rows, conf_value):
    conf_label = f"{conf_value:g}"
    headers = [
        "name",
        "mAP50",
        "mAP50-95",
        f"Precision (mean at conf={conf_label})",
        f"Recall (mean at conf={conf_label})",
        "mAP50-95 for Class 0",
        "mAP50-95 for Class 1",
        f"Precision at conf={conf_label} for Class0",
        f"Recall at conf={conf_label} for Class0",
        f"Precision at conf={conf_label} for Class1",
        f"Recall at conf={conf_label} for Class1",
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
            f"{row['precision_class0']:.6f}",
            f"{row['recall_class0']:.6f}",
            f"{row['precision_class1']:.6f}",
            f"{row['recall_class1']:.6f}",
        ])

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

    results_dir = "results"
    eval_results_dir = os.path.join(results_dir, OUTPUT_DIR)
    os.makedirs(eval_results_dir, exist_ok=True)
    csv_path = os.path.join(eval_results_dir, f"{MODELS}_eval_.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_rows)
    print(f"\nSummary CSV saved to: {csv_path}")

# =========================
# 评估函数（只用 val）
# =========================
def evaluate_model(model_path, data_yaml):
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    print(f"\n===== {model_name} =====")

    model = YOLO(model_path)

    metrics = model.val(
        data=data_yaml,
        conf=VAL_CONF,
        split=DATASET,
        iou=VAL_IOU,
        plots=True,        # confusion matrix + PR curve
        save_json=True,
        verbose=True,
        project=OUTPUT_DIR,
        name=model_name,
        exist_ok=True
    )

    # =========================
    # 输出指标
    # =========================
    print("\n===== METRICS =====")

    print("mAP50:", float(metrics.box.map50))
    print("mAP50-95:", float(metrics.box.map))

    print("Precision (mean):", float(metrics.box.mp))
    print("Recall (mean):", float(metrics.box.mr))

    map_class0 = _safe_float(metrics.box.maps[0]) if metrics.box.maps is not None and len(metrics.box.maps) > 0 else 0.0
    map_class1 = _safe_float(metrics.box.maps[1]) if metrics.box.maps is not None and len(metrics.box.maps) > 1 else 0.0
    precision_class0, recall_class0 = _get_class_pr(metrics, 0)
    precision_class1, recall_class1 = _get_class_pr(metrics, 1)

    # =========================
    # per-class mAP
    # =========================
    if metrics.box.maps is not None:
        print("\nPer-class mAP:")
        for i, v in enumerate(metrics.box.maps):
            print(f"Class {i}: {float(v)}")
            
    # =========================
    # per-class Precision & Recall
    # =========================
    if metrics.box.p is not None and metrics.box.r is not None:
        print("\nPer-class Precision & Recall:")
        for i in range(len(metrics.box.p)):
            p = float(metrics.box.p[i])
            r = float(metrics.box.r[i])
            print(f"Class {i}: Precision={p}, Recall={r}")
            
    # =========================
    # confusion matrix
    # =========================
    if hasattr(metrics, "confusion_matrix"):
        cm = metrics.confusion_matrix.matrix
        print("\nConfusion Matrix:\n", cm)

    # =========================
    # 保存路径提示
    # =========================
    print("\nResults saved in:")
    print(os.path.join("runs", "detect", OUTPUT_DIR, model_name))

    # 清理
    del model

    return {
        "name": model_name,
        "mAP50": _safe_float(metrics.box.map50),
        "mAP50_95": _safe_float(metrics.box.map),
        "precision_mean": _safe_float(metrics.box.mp),
        "recall_mean": _safe_float(metrics.box.mr),
        "map_class0": map_class0,
        "map_class1": map_class1,
        "precision_class0": precision_class0,
        "recall_class0": recall_class0,
        "precision_class1": precision_class1,
        "recall_class1": recall_class1,
    }

# =========================
# 主函数
# =========================
if __name__ == "__main__":

    if not os.path.exists(DATA_YAML):
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    summary_rows = []
    for m in MODEL_PATHS:
        summary_rows.append(evaluate_model(m, DATA_YAML))

    if summary_rows:
        print_summary_table(summary_rows, VAL_CONF)
