import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from joblib import dump, load
try:
    from rich.console import Console
    from rich.table import Table
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

# ===========================
# 小工具：Specificity / NPV
# ===========================
def specificity_npv(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        return np.nan, np.nan
    tn, fp, fn, tp = cm.ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    npv  = tn / (tn + fn) if (tn + fn) > 0 else np.nan
    return spec, npv

# ===========================
# Pretty table printer
# ===========================
def pretty_print_table(df, title=None, float_cols=None, float_digits=4):
    if float_cols is None:
        float_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df_show = df.copy()
    for c in float_cols:
        df_show[c] = df_show[c].astype(float).round(float_digits)

    if _HAS_RICH:
        console = Console()
        if title:
            console.rule(f"[bold]{title}")
        table = Table(show_header=True, header_style="bold", box=None, pad_edge=False)
        for c in df_show.columns:
            align = "right" if c in float_cols else "left"
            table.add_column(str(c), justify=align, no_wrap=True)
        for _, row in df_show.iterrows():
            row_vals = []
            for c in df_show.columns:
                v = row[c]
                if pd.isna(v):
                    row_vals.append("-")
                elif c in float_cols:
                    row_vals.append(f"{float(v):.{float_digits}f}")
                else:
                    row_vals.append(str(v))
            table.add_row(*row_vals)
        console.print(table)
    else:
        if title: print(f"\n--- {title} ---")
        print(df_show.to_string(index=False))

# ===========================
# 模型保存與載入
# ===========================
def save_best_model(models_dir, label, model_obj, scaler, imputer, feature_columns, outlier_bounds, threshold):
    os.makedirs(models_dir, exist_ok=True)
    base = f"{label}_best"
    model_path   = os.path.join(models_dir, base + ".joblib")
    scaler_path  = os.path.join(models_dir, base + "_scaler.joblib")
    imputer_path = os.path.join(models_dir, base + "_imputer.joblib")
    meta_path    = os.path.join(models_dir, base + ".json")

    dump(model_obj, model_path)
    if scaler is not None: dump(scaler, scaler_path)
    if imputer is not None: dump(imputer, imputer_path)

    meta = {
        "label": label,
        "threshold": float(threshold),
        "feature_columns": list(feature_columns),
        "outlier_bounds": outlier_bounds,
        "files": {
            "model": os.path.basename(model_path),
            "scaler": os.path.basename(scaler_path) if scaler is not None else None,
            "imputer": os.path.basename(imputer_path) if imputer is not None else None,
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存整體最佳模型：{model_path}")

def load_best_model_and_meta(models_dir, label):
    base = f"{label}_best"
    model_path = os.path.join(models_dir, base + ".joblib")
    scaler_path = os.path.join(models_dir, base + "_scaler.joblib")
    imputer_path = os.path.join(models_dir, base + "_imputer.joblib")
    meta_path = os.path.join(models_dir, base + ".json")

    if not os.path.isfile(model_path) or not os.path.isfile(meta_path):
        return None

    model = load(model_path)
    scaler = load(scaler_path) if os.path.isfile(scaler_path) else None
    imputer = load(imputer_path) if os.path.isfile(imputer_path) else None

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return {
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "threshold": meta.get("threshold", 0.5),
        "feature_columns": meta.get("feature_columns", []),
        "outlier_bounds": meta.get("outlier_bounds", {}),
    }