# train_gad_vs_health_stacking.py
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import BorderlineSMOTE

# [修改點 1] 改為引用 ProcessorData2Full (因為它是包含所有特徵的版本)
from processors import ProcessorData2Full
from utils import pretty_print_table

# ================= 設定區 =================
FILE_PATH = r"D:\ML_Project\dataset\data.xlsx" # 請確認你的路徑是否正確
SHEET_NAME = "Data2"
TARGET_LABEL = "GAD"
N_FOLDS = 10
RANDOM_SEED = 42
# ==========================================

def create_interaction_features(df, hrv_cols, psych_cols):
    """
    建立特徵交互作用 (包含 Sex, BMI, Age, Psych, HRV)
    """
    df_new = df.copy()

    # 1. HRV * Psych
    for h_col in hrv_cols:
        if h_col not in df.columns: continue
        for p_col in psych_cols:
            if p_col not in df.columns: continue
            df_new[f"Inter_{h_col}_x_{p_col}"] = df[h_col] * df[p_col]

    # 2. Age * HRV
    if "Age" in df.columns:
        for h_col in hrv_cols:
            if h_col not in df.columns: continue
            df_new[f"Inter_Age_{h_col}"] = df["Age"] * df[h_col]

    # 3. BMI * HRV
    if "BMI" in df.columns:
        for h_col in hrv_cols:
            if h_col not in df.columns: continue
            df_new[f"Inter_BMI_{h_col}"] = df["BMI"] * df[h_col]
            
    # 4. Sex * Psych
    if "Sex" in df.columns:
        for p_col in psych_cols:
            if p_col not in df.columns: continue
            df_new[f"Inter_Sex_{p_col}"] = df["Sex"] * df[p_col]

    return df_new

def add_deviation_features(X_train, X_val, y_train, hrv_cols, extra_cols=['BMI']):
    """
    相對偏差特徵 (Z-score based on Healthy Control Group)
    """
    X_tr_new = X_train.copy()
    X_val_new = X_val.copy()

    # y_train == 0 在這裡明確代表 Health Group
    neg_mask = (y_train == 0)
    X_neg = X_train[neg_mask]
    
    target_cols = [c for c in hrv_cols if c in X_train.columns]
    for c in extra_cols:
        if c in X_train.columns:
            target_cols.append(c)

    for col in target_cols:
        if len(X_neg) > 1:
            mu = X_neg[col].mean()
            sigma = X_neg[col].std() + 1e-6
        else:
            mu = X_train[col].mean()
            sigma = X_train[col].std() + 1e-6

        feat_name = f"Dev_Z_{col}"
        X_tr_new[feat_name] = (X_train[col] - mu) / sigma
        X_val_new[feat_name] = (X_val[col] - mu) / sigma

    return X_tr_new, X_val_new

def get_specialized_model(pos_scale_weight):
    """
    Stacking 模型架構
    """
    adjusted_weight = max(pos_scale_weight * 0.9, 1e-3)

    # 1. XGBoost
    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.03,
        scale_pos_weight=adjusted_weight,
        gamma=1.0,
        reg_lambda=10.0,
        eval_metric="auc",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbosity=0
    )
    
    # 2. LightGBM
    lgbm_clf = lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=20,
        learning_rate=0.03,
        class_weight="balanced",
        reg_lambda=10.0,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=-1
    )
    
    # 3. Random Forest
    rf_clf = RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        class_weight="balanced",
        min_samples_leaf=4,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    calibrated_xgb = CalibratedClassifierCV(xgb_clf, method='isotonic', cv=3)
    calibrated_lgbm = CalibratedClassifierCV(lgbm_clf, method='sigmoid', cv=3)

    ensemble = StackingClassifier(
        estimators=[
            ('xgb', calibrated_xgb), 
            ('lgbm', calibrated_lgbm), 
            ('rf', rf_clf)
        ],
        final_estimator=LogisticRegression(class_weight='balanced', C=1.0, random_state=RANDOM_SEED),
        stack_method='predict_proba',
        n_jobs=-1
    )
    return ensemble

def main():
    print("🚀 啟動 GAD vs Health 專用版 (Stacking)")
    print("🎯 目標: 區分 GAD 患者與健康受試者 (排除其他共病干擾)")

    # 1. Load Data
    # [修改點 2] 使用 ProcessorData2Full 並移除 mode 參數
    processor = ProcessorData2Full(FILE_PATH, SHEET_NAME)
    
    if not processor.load_data(): return
    if not processor.prepare_features_and_labels(): return

    # ================= 數據過濾邏輯 =================
    df_full = processor.df
    X_full = processor.X
    
    if 'Health' not in df_full.columns:
        print("❌ Error: 'Health' column not found in dataset. Cannot filter for GAD vs Health.")
        return

    # 建立過濾 Mask
    mask_gad = df_full[TARGET_LABEL] == 1
    mask_health = (df_full['Health'] == 1) & (df_full[TARGET_LABEL] == 0)
    mask_valid = mask_gad | mask_health
    
    # 套用過濾
    X_raw = X_full.loc[mask_valid].copy().reset_index(drop=True)
    y_raw = df_full.loc[mask_valid, TARGET_LABEL].copy().reset_index(drop=True)
    
    n_gad = mask_gad.sum()
    n_health = mask_health.sum()
    n_filtered = len(df_full) - len(X_raw)
    
    print(f"\n📊 數據篩選統計:")
    print(f"   - GAD Samples: {n_gad}")
    print(f"   - Health Samples: {n_health}")
    print(f"   - Excluded (Non-GAD & Non-Health): {n_filtered}")
    print(f"   - Final Dataset: {X_raw.shape}, Positive Rate: {y_raw.mean():.2%}")
    # ==========================================================

    # 2. 特徵清單
    hrv_cols = ['SDNN', 'TP', 'LF', 'HF', 'LFHF', 'VLF', 'NLF', 'MEANH']
    psych_cols = ['phq15', 'haq21', 'bdi', 'bai', 'cabah']
    
    # 3. CV Loop
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    metrics_list = []
    
    fold = 1
    for train_idx, val_idx in skf.split(X_raw, y_raw):
        print(f"\n📂 Fold {fold}/{N_FOLDS}")

        X_train = X_raw.iloc[train_idx].copy()
        X_val = X_raw.iloc[val_idx].copy()
        y_train = y_raw.iloc[train_idx]
        y_val = y_raw.iloc[val_idx]

        # A. Impute
        num_cols = X_train.select_dtypes(include=[np.number]).columns
        imputer = KNNImputer(n_neighbors=5)
        X_train[num_cols] = imputer.fit_transform(X_train[num_cols])
        X_val[num_cols] = imputer.transform(X_val[num_cols])

        # B. Feature Engineering
        X_train = create_interaction_features(X_train, hrv_cols, psych_cols)
        X_val = create_interaction_features(X_val, hrv_cols, psych_cols)
        X_train, X_val = add_deviation_features(X_train, X_val, y_train, hrv_cols, extra_cols=['BMI'])

        # C. Scale
        scaler = RobustScaler()
        X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_s = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)

        # D. Sampling 
        smote = BorderlineSMOTE(sampling_strategy=0.50, k_neighbors=5, random_state=RANDOM_SEED)
        try:
            X_res, y_res = smote.fit_resample(X_train_s, y_train)
        except:
            X_res, y_res = X_train_s, y_train

        # E. Feature Selection
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1),
            threshold="0.5*mean"
        )
        selector.fit(X_res, y_res)
        
        selected_mask = selector.get_support()
        selected_feats = X_res.columns[selected_mask]
        
        X_res_selected = X_res[selected_feats].copy()
        X_val_selected = X_val_s[selected_feats].copy()
        
        print(f"   ✂️ 特徵篩選: {X_res.shape[1]} -> {X_res_selected.shape[1]} features selected")

        # F. Train (Stacking)
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum() if y_train.sum() > 0 else 1
        model = get_specialized_model(pos_weight)
        model.fit(X_res_selected, y_res)

        # G. Predict
        y_pred_proba = model.predict_proba(X_val_selected)[:, 1]
        auc_score = roc_auc_score(y_val, y_pred_proba)

        # Threshold Optimization
        thresholds = np.linspace(0.1, 0.9, 100)
        best_f1, best_th = 0.0, 0.5
        for th in thresholds:
            pred = (y_pred_proba >= th).astype(int)
            f1 = f1_score(y_val, pred, zero_division=0)
            if f1 > best_f1: best_f1, best_th = f1, th

        y_pred = (y_pred_proba >= best_th).astype(int)
        acc = (y_pred == y_val).mean()
        
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        rec = recall_score(y_val, y_pred, zero_division=0)
        prec = precision_score(y_val, y_pred, zero_division=0)

        print(f"   -> AUC: {auc_score:.4f} | Spec: {spec:.3f} | Sens: {rec:.3f}")
        
        metrics_list.append({
            "Fold": fold, "AUC": auc_score, "F1": best_f1, "Acc": acc,
            "Sens(Recall)": rec, "Spec": spec, "Prec": prec, "NPV": npv
        })
        fold += 1

    # Summary
    df_res = pd.DataFrame(metrics_list)
    avg_auc = df_res["AUC"].mean()
    
    print("\n" + "=" * 50)
    print(f"🏆 Final Result (GAD vs Health Stacking)")
    print("=" * 50)
    pretty_print_table(df_res, title="Per Fold Metrics")

    summary = df_res.mean(numeric_only=True).to_frame().T
    summary["Fold"] = "Average"
    pretty_print_table(summary)

    print(f"\n✨ Final AUC = {avg_auc:.4f}")

if __name__ == "__main__":
    main()