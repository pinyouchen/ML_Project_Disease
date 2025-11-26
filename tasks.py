import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc, 
    f1_score, accuracy_score, precision_score, 
    recall_score, confusion_matrix, roc_auc_score
)
from sklearn.ensemble import IsolationForest

# 引用自定義模組
from model_trainer import ModelTrainer
from utils import save_best_model, pretty_print_table, load_best_model_and_meta
from visualization import Visualizer
import shap 

# 引用 Processors
from processors import (
    ProcessorHRV, 
    ProcessorPsych, 
    ProcessorBaselineAll, 
    ProcessorFullV62, 
    DataProcessorBaseline
)

def run_binary_task(task_name, file_path, sheet_name, processor_cls, use_stacking=True):
    print("\n" + "="*70)
    print(f"執行任務: {task_name} (AutoML & SHAP-OOF Version)")
    print("="*70)
    
    timestamp = datetime.now().strftime(f"{task_name}_%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.getcwd(), "runs1", timestamp)
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. 載入並準備數據
    processor = processor_cls(file_path, sheet_name)
    if not processor.load_data(): return
    if not processor.prepare_features_and_labels(): return
    
    label_names = ['SSD', 'MDD', 'Panic', 'GAD']
    y_dict = processor.y_dict
    df_full = processor.df
    X_full = processor.X
    
    summary_rows = []
    
    # 全域比較容器
    overall_roc_data = {}
    overall_pr_data = {}
    overall_metrics_list = [] 
    
    for label in label_names:
        if label not in y_dict: continue
        print(f"\n🩺 診斷：{label} vs Health")
        
        # 2. 定義正負樣本 (Disease vs Health Only)
        mask_disease = df_full[label] == 1
        mask_health = (df_full['Health'] == 1) & (df_full[label] == 0)
        mask_valid = mask_disease | mask_health
        X_sub = X_full.loc[mask_valid].copy()
        y_sub = np.where(mask_disease[mask_valid], 1, 0)
        
        # 初始化視覺化物件
        viz = Visualizer(label, run_dir, sub_folder=label)

        # [NEW] 繪製 Correlation Matrix (在 CV 之前)
        # 為了畫圖，我們先做一次整體的 Impute (fit=True)
        # 注意：這只是為了畫 EDA 圖，不會用於後續訓練 (訓練會重新在 Fold 內處理)
        print("   📊 繪製特徵相關性矩陣 (EDA)...")
        try:
            X_sub_corr_p = processor.impute_and_scale(X_sub, fit=True)
            viz.plot_correlation_matrix(X_sub_corr_p)
        except Exception as e:
            print(f"   ⚠️ 無法繪製 Correlation Matrix: {e}")

        # 設定目標 F1 (僅供 log 顯示參考)
        base_f1 = {'SSD':0.66, 'MDD':0.46, 'Panic':0.50, 'GAD':0.57}.get(label, 0.5)
        target_f1 = {'SSD':0.75, 'MDD':0.75, 'Panic':0.55, 'GAD':0.70}.get(label, 0.7)
        
        # 3. 外部迴圈 (Outer CV Loop)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        metrics_list = []
        tprs = []; mean_fpr = np.linspace(0, 1, 100)
        roc_aucs = []
        precisions = []; mean_recall = np.linspace(0, 1, 100)
        pr_aucs = []
        no_skill = y_sub.sum() / len(y_sub)
        
        y_true_all = []; y_pred_all = []
        
        # [NEW] SHAP 收集容器 (用於 OOF 串接)
        shap_values_list = []
        X_test_shap_list = []
        importance_list = []
        
        best_model_info = {"f1": -1.0, "p": -1.0, "obj": None, "name": None}
        
        fold_id = 1
        for train_idx, test_idx in skf.split(X_sub, y_sub):
            print(f"\n   📂 Fold {fold_id}/5")
            X_tr, X_te = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
            
            X_tr = X_tr.reset_index(drop=True)
            X_te = X_te.reset_index(drop=True)
            y_tr = pd.Series(y_sub[train_idx]) 
            y_te = pd.Series(y_sub[test_idx])
            
            # 4. 數據前處理 (Inside Fold -> No Leakage)
            X_tr_p, X_te_p = processor.impute_and_scale(X_tr, X_te, fit=True)
            
            # 5. Isolation Forest
            iso = IsolationForest(contamination=0.03, random_state=42, n_jobs=1)
            outlier_preds = iso.fit_predict(X_tr_p)
            mask_clean = (outlier_preds == 1) | (y_tr == 1)
            X_tr_clean = X_tr_p[mask_clean].copy()
            y_tr_clean = y_tr[mask_clean].copy()
            
            removed = len(X_tr_p) - len(X_tr_clean)
            if removed > 0: print(f"      🧹 移除了 {removed} 個異常樣本")

            # 6. 訓練 (AutoML)
            trainer = ModelTrainer(label, y_tr_clean.sum(), len(y_tr_clean)-y_tr_clean.sum(), base_f1, target_f1, use_stacking)
            trainer.build_models()
            res = trainer.train_and_evaluate(X_tr_clean, X_te_p, y_tr_clean, y_te)
            
            # 7. SHAP 收集
            # 優先從樹模型 (XGB/LGBM) 提取解釋
            target_shap_model = None
            if 'XGB' in trainer.fitted_models: target_shap_model = trainer.fitted_models['XGB']
            elif 'LGBM' in trainer.fitted_models: target_shap_model = trainer.fitted_models['LGBM']
            
            if target_shap_model:
                try:
                    explainer = shap.TreeExplainer(target_shap_model)
                    shap_vals = explainer.shap_values(X_te_p)
                    
                    # 處理不同套件回傳格式差異 (List vs Array)
                    if isinstance(shap_vals, list) and len(shap_vals) == 2:
                        # Binary Case:取 class 1
                        shap_values_list.append(shap_vals[1]) 
                    elif isinstance(shap_vals, np.ndarray):
                        # 可能是 (samples, features) 或 (samples, features, classes)
                        if shap_vals.ndim == 2:
                            shap_values_list.append(shap_vals)
                        elif shap_vals.ndim == 3: # LGBM 有時會這樣
                            shap_values_list.append(shap_vals[:, :, 1])
                    
                    X_test_shap_list.append(X_te_p)
                    
                    if hasattr(target_shap_model, 'feature_importances_'):
                        importance_list.append(pd.DataFrame({
                            'Feature': X_te_p.columns, 'Importance': target_shap_model.feature_importances_
                        }))
                except Exception as e:
                    # 不中斷流程
                    pass

            # 8. 選擇當折最佳模型 (用於 Metrics)
            special = [m for m in res.keys() if m in ['Ensemble', 'Stacking']]
            show_name = max(special, key=lambda k: res[k]['f1_score']) if special else max(res.keys(), key=lambda k: res[k]['f1_score'])
            r = res[show_name]
            
            metrics_list.append({
                'F1': r['f1_score'], 'Acc': r['accuracy'], 'AUC': r['auc'],
                'Prec': r['precision'], 'Recall': r['recall'], 
                'Spec': r['specificity'], 'NPV': r['npv']
            })
            
            fpr, tpr, _ = roc_curve(y_te, r['y_pred_proba'])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            roc_aucs.append(r['auc'])
            
            prec, rec, _ = precision_recall_curve(y_te, r['y_pred_proba'])
            precisions.append(np.interp(mean_recall, rec[::-1], prec[::-1]))
            pr_aucs.append(auc(rec, prec))
            
            y_true_all.extend(y_te)
            y_pred_all.extend(r['y_pred'])
            
            # 9. 更新整體最佳模型 (僅保存 Single Model)
            singles = [k for k in res.keys() if k not in ['Ensemble', 'Stacking']]
            if singles:
                best_s_name = max(singles, key=lambda k: res[k]['f1_score'])
                best_s = res[best_s_name]
                is_better = False
                if best_s['f1_score'] > best_model_info['f1']: is_better = True
                elif best_s['f1_score'] == best_model_info['f1'] and best_s['precision'] > best_model_info['p']: is_better = True
                
                if is_better:
                    best_model_info = {
                        "f1": best_s['f1_score'], "p": best_s['precision'],
                        "obj": best_s['model'], "name": best_s_name, "fold": fold_id,
                        "thresh": best_s['threshold'], 
                        "scaler": processor.scaler,
                        "imputer": processor.knn_imputer, 
                        "cols": X_tr_p.columns,
                        "bounds": processor.outlier_bounds_,
                        "r": best_s['recall'], "spec": best_s['specificity'], 
                        "npv": best_s['npv'], "auc": best_s['auc'], "acc": best_s['accuracy']
                    }
            fold_id += 1

        # 10. 視覺化 (Loop 結束後)
        X_sub_p = processor.impute_and_scale(X_sub, fit=True)
        viz.plot_pca_scatter(X_sub_p, y_sub)

        df_metrics_fold = pd.DataFrame(metrics_list)
        metrics_summary = df_metrics_fold.mean().to_dict()
        metrics_std = df_metrics_fold.std().to_dict()
        
        df_bar = pd.DataFrame({
            'Metric': list(metrics_summary.keys()),
            'Mean': list(metrics_summary.values()),
            'Std': list(metrics_std.values())
        })
        viz.plot_performance_metrics(df_bar)
        
        for m, v in metrics_summary.items(): 
            overall_metrics_list.append({'Label': label, 'Metric': m, 'Mean': v, 'Std': metrics_std[m]})
        
        viz.plot_roc_curve_with_ci(tprs, mean_fpr, roc_aucs)
        mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
        overall_roc_data[label] = (mean_fpr, mean_tpr, np.mean(roc_aucs))
        
        viz.plot_pr_curve_with_ci(precisions, mean_recall, pr_aucs, no_skill)
        mean_prec = np.mean(precisions, axis=0)
        overall_pr_data[label] = (mean_recall, mean_prec, np.mean(pr_aucs))
        
        viz.plot_confusion_matrix_aggregated(y_true_all, y_pred_all)
        viz.plot_radar_chart({k: metrics_summary[k] for k in ['F1', 'Acc', 'Prec', 'Recall', 'Spec', 'AUC']})
        
        if importance_list: 
            viz.plot_feature_importance_boxplot(pd.concat(importance_list))
            
        # [NEW] 繪製 Global OOF SHAP
        # 串接 5 個 Fold 的 SHAP 與 Feature Data
        if shap_values_list and X_test_shap_list:
            try:
                # 簡單檢查長度是否一致
                if len(shap_values_list) == len(X_test_shap_list):
                    global_shap_values = np.concatenate(shap_values_list, axis=0)
                    global_X_test = pd.concat(X_test_shap_list, axis=0)
                    
                    # 確保沒有形狀不匹配 (例如其中一個 fold 是空的)
                    if global_shap_values.shape[0] == global_X_test.shape[0]:
                        print(f"   📊 繪製 Global OOF SHAP Summary (N={global_X_test.shape[0]})...")
                        viz.plot_shap_summary_oof(global_shap_values, global_X_test)
                else:
                    print("   ⚠️ SHAP 列表長度不一致，跳過繪圖")
            except Exception as e: 
                print(f"   ⚠️ SHAP Plot Error: {e}")

        # 保存模型
        if best_model_info['obj']:
            save_best_model(
                models_dir, label, best_model_info['obj'], 
                best_model_info['scaler'], best_model_info['imputer'],
                best_model_info['cols'], best_model_info['bounds'], 
                best_model_info['thresh']
            )

        summary_rows.append({
            "Label": label, "BestModel": best_model_info['name'],
            "F1(Best)": best_model_info['f1'], "P(Best)": best_model_info.get('p', 0),
            "R(Best)": best_model_info.get('r', 0), "Spec(Best)": best_model_info.get('spec', 0),
            "NPV(Best)": best_model_info.get('npv', 0), "AUC(Best)": best_model_info.get('auc', 0),
            "ACC(Best)": best_model_info.get('acc', 0),
            "F1(avg)": metrics_summary['F1'], "P(avg)": metrics_summary['Prec'],
            "R(avg)": metrics_summary['Recall'], "Spec(avg)": metrics_summary['Spec'],
            "NPV(avg)": metrics_summary['NPV'], "AUC(avg)": metrics_summary['AUC'],
            "ACC(avg)": metrics_summary['Acc']
        })

    print("\n📊 正在繪製多疾病比較總圖...")
    viz_summary = Visualizer("Comparison", run_dir, sub_folder="Summary_Comparison")
    if overall_metrics_list: viz_summary.plot_multilabel_metrics(pd.DataFrame(overall_metrics_list))
    if overall_roc_data: viz_summary.plot_multilabel_roc(overall_roc_data)
    if overall_pr_data: viz_summary.plot_multilabel_pr(overall_pr_data)

    if summary_rows:
        res_df = pd.DataFrame(summary_rows)
        res_df.to_excel(os.path.join(run_dir, "Results_Summary.xlsx"), index=False)
        cols = ["Label", "BestModel", 
                "F1(Best)", "P(Best)", "R(Best)", "Spec(Best)", "NPV(Best)", "AUC(Best)", "ACC(Best)", 
                "F1(avg)", "P(avg)", "R(avg)", "Spec(avg)", "NPV(avg)", "AUC(avg)", "ACC(avg)"]
        pretty_print_table(res_df[cols], title="最終結果摘要")
        print(f"\n所有結果與比較圖表已存至: {run_dir}/plots")


# def run_external_validation(models_dir_input, file_path, sheet_name, processor_cls=DataProcessorBaseline):
#     """
#     外部驗證函式
#     更新：增加 processor_cls 參數，允許傳入 FullV62 等高階處理器
#     """
#     print("\n" + "="*70)
#     print(f"執行外部驗證 (Data1)")
#     print(f"模型來源: {models_dir_input}")
#     print(f"使用處理器: {processor_cls.__name__}")
#     print("="*70)
    
#     if not os.path.exists(models_dir_input):
#         print(f"找不到模型資料夾: {models_dir_input}")
#         return

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     val_out_dir = os.path.join(os.path.dirname(models_dir_input), f"Validation_Data1_{timestamp}")
#     os.makedirs(val_out_dir, exist_ok=True)

#     # 載入數據
#     processor = processor_cls(file_path, sheet_name)
    
#     if not processor.load_data(): return
#     if not processor.prepare_features_and_labels(): return
    
#     df_full = processor.df
#     X_full = processor.X
#     label_names = ['SSD', 'MDD', 'Panic', 'GAD']
    
#     results = []
    
#     for label in label_names:
#         print(f"\n🔍 驗證：{label}")
        
#         # 載入模型 Metadata
#         info = load_best_model_and_meta(models_dir_input, label)
#         if not info:
#             print(f"  無法載入 {label} 模型，跳過")
#             continue
            
#         # 篩選資料 (Data1 的結構)
#         if label not in df_full.columns or 'Health' not in df_full.columns: continue
        
#         mask_valid = (
#             df_full['Health'].isin([0, 1]) &
#             df_full[label].isin([0, 1]) &
#             ((df_full['Health'] == 1) | (df_full[label] == 1))
#         )
#         df_sub = df_full.loc[mask_valid].copy()
#         X_sub = X_full.loc[mask_valid].copy()
        
#         mask_xor = (df_sub['Health'] == 1) ^ (df_sub[label] == 1)
#         df_sub = df_sub.loc[mask_xor]
#         X_sub = X_sub.loc[mask_xor]
        
#         y_sub = np.where(df_sub[label] == 1, 1, 0)
        
#         if len(y_sub) == 0:
#             print("  無有效樣本")
#             continue
        
#         print(f"   樣本數: {len(y_sub)} (正例={y_sub.sum()}, 負例={len(y_sub)-y_sub.sum()})")

#         # [Check] 對齊特徵：非常重要！確保驗證集的特徵與訓練集完全一致
#         required_cols = info['feature_columns']
#         X_eval = pd.DataFrame(index=X_sub.index)
        
#         missing_cols = []
#         for col in required_cols:
#             if col in X_sub.columns:
#                 X_eval[col] = X_sub[col]
#             else:
#                 X_eval[col] = np.nan
#                 missing_cols.append(col)
        
#         if missing_cols:
#             print(f"   ⚠️ 警告：驗證集缺少以下訓練特徵 (將補 NaN): {missing_cols[:5]}...")
        
#         X_eval = X_eval[required_cols]
            
#         # 套用訓練時的 Preprocessor 狀態 (Scaler, Imputer, Bounds)
#         processor.outlier_bounds_ = info['outlier_bounds']
#         processor.knn_imputer = info['imputer']
#         processor.scaler = info['scaler']
        
#         # 使用 fit=False，確保完全依照訓練集的參數轉換
#         X_eval_p = processor.impute_and_scale(X_eval, fit=False)
        
#         # 推論
#         model = info['model']
#         try:
#             proba = model.predict_proba(X_eval_p)[:, 1]
#         except:
#             proba = model.predict(X_eval_p)
        
#         # 使用模型訓練時存下來的最佳閾值 (這個閾值已經是經過 CV 驗證的)
#         threshold = info['threshold']
#         pred = (proba >= threshold).astype(int)
        
#         f1 = f1_score(y_sub, pred)
#         acc = accuracy_score(y_sub, pred)
#         prec = precision_score(y_sub, pred, zero_division=0)
#         rec = recall_score(y_sub, pred, zero_division=0)
        
#         cm = confusion_matrix(y_sub, pred, labels=[0, 1])
#         tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
#         spec = tn/(tn+fp) if (tn+fp)>0 else 0
#         npv = tn/(tn+fn) if (tn+fn)>0 else 0
        
#         try: auc_val = roc_auc_score(y_sub, proba)
#         except: auc_val = np.nan
        
#         print(f"   → Result: F1={f1:.4f}, Acc={acc:.4f}, AUC={auc_val:.4f}, Spec={spec:.4f} (Th={threshold:.2f})")
        
#         results.append({
#             "Label": label, "F1": f1, "Acc": acc, "AUC": auc_val, 
#             "Spec": spec, "NPV": npv, "Prec": prec, "Recall": rec,
#             "Threshold": threshold
#         })
        
#         # 驗證集繪圖
#         viz = Visualizer(label, val_out_dir, sub_folder=label)
#         viz.plot_confusion_matrix_aggregated(y_sub, pred)
#         if len(np.unique(y_sub)) > 1:
#             fpr, tpr, _ = roc_curve(y_sub, proba)
#             viz.plot_roc_curve_with_ci([tpr], fpr, [auc_val])
            
#     if results:
#         res_df = pd.DataFrame(results)
#         excel_path = os.path.join(val_out_dir, "External_Validation_Results.xlsx")
#         res_df.to_excel(excel_path, index=False)
        
#         cols = ["Label", "F1", "Prec", "Recall", "Spec", "NPV", "AUC", "Acc", "Threshold"]
        
#         pretty_print_table(res_df[cols], title="外部驗證結果摘要")
#         print(f"\n外部驗證完成，結果已儲存至: {val_out_dir}")
#     else:
#         print("\n無法產生任何驗證結果 (可能缺資料或模型)")

def run_external_validation(models_dir_input, file_path, sheet_name, processor_cls=DataProcessorBaseline):
    """
    外部驗證函式 (含閾值診斷功能)
    """
    print("\n" + "="*70)
    print(f"執行外部驗證 (Data1)")
    print(f"模型來源: {models_dir_input}")
    print(f"使用處理器: {processor_cls.__name__}")
    print("="*70)
    
    if not os.path.exists(models_dir_input):
        print(f"找不到模型資料夾: {models_dir_input}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    val_out_dir = os.path.join(os.path.dirname(models_dir_input), f"Validation_Data1_{timestamp}")
    os.makedirs(val_out_dir, exist_ok=True)

    # 1. 載入數據
    processor = processor_cls(file_path, sheet_name)
    
    if not processor.load_data(): return
    if not processor.prepare_features_and_labels(): return
    
    df_full = processor.df
    X_full = processor.X
    label_names = ['SSD', 'MDD', 'Panic', 'GAD']
    
    results = []
    
    for label in label_names:
        print(f"\n🔍 驗證：{label}")
        
        # 2. 載入模型 Metadata
        info = load_best_model_and_meta(models_dir_input, label)
        if not info:
            print(f"  無法載入 {label} 模型，跳過")
            continue
            
        # 3. 篩選資料 (Data1 的結構)
        if label not in df_full.columns or 'Health' not in df_full.columns: continue
        
        mask_valid = (
            df_full['Health'].isin([0, 1]) &
            df_full[label].isin([0, 1]) &
            ((df_full['Health'] == 1) | (df_full[label] == 1))
        )
        df_sub = df_full.loc[mask_valid].copy()
        X_sub = X_full.loc[mask_valid].copy()
        
        # 確保互斥 (Health=1 xor Disease=1)
        mask_xor = (df_sub['Health'] == 1) ^ (df_sub[label] == 1)
        df_sub = df_sub.loc[mask_xor]
        X_sub = X_sub.loc[mask_xor]
        
        y_sub = np.where(df_sub[label] == 1, 1, 0)
        
        if len(y_sub) == 0:
            print("  無有效樣本")
            continue
        
        print(f"   樣本數: {len(y_sub)} (正例={y_sub.sum()}, 負例={len(y_sub)-y_sub.sum()})")

        # 4. 對齊特徵 (Feature Alignment)
        required_cols = info['feature_columns']
        X_eval = pd.DataFrame(index=X_sub.index)
        
        missing_cols = []
        for col in required_cols:
            if col in X_sub.columns:
                X_eval[col] = X_sub[col]
            else:
                X_eval[col] = np.nan
                missing_cols.append(col)
        
        if missing_cols:
            print(f"   ⚠️ 警告：驗證集缺少以下訓練特徵 (將補 NaN): {missing_cols[:5]}...")
        
        X_eval = X_eval[required_cols]
            
        # 5. 套用訓練時的 Preprocessor (Scaler, Imputer, Bounds)
        processor.outlier_bounds_ = info['outlier_bounds']
        processor.knn_imputer = info['imputer']
        processor.scaler = info['scaler']
        
        # fit=False 確保使用訓練集的統計參數
        X_eval_p = processor.impute_and_scale(X_eval, fit=False)
        
        # 6. 推論
        model = info['model']
        try:
            proba = model.predict_proba(X_eval_p)[:, 1]
        except:
            proba = model.predict(X_eval_p) # 若模型不支援機率，回退到類別
        
        try: auc_val = roc_auc_score(y_sub, proba)
        except: auc_val = np.nan

        # ==========================================
        # [新增功能] 多閾值診斷 (Threshold Diagnostic)
        # ==========================================
        original_threshold = info['threshold']
        
        # 定義我們要測試的閾值：包含原本的，以及較高的幾個選項
        test_candidates = [original_threshold, 0.4, 0.5, 0.6, 0.7]
        test_thresholds = sorted(list(set(test_candidates))) # 排序並去重
        
        print(f"   📊 閾值敏感度測試 (原本 Th={original_threshold:.3f})...")
        
        final_stats = {} # 用來存原本閾值的結果 (寫入 Excel 用)

        for th in test_thresholds:
            pred_th = (proba >= th).astype(int)
            
            # 計算指標
            f1_th = f1_score(y_sub, pred_th)
            acc_th = accuracy_score(y_sub, pred_th)
            prec_th = precision_score(y_sub, pred_th, zero_division=0)
            rec_th = recall_score(y_sub, pred_th, zero_division=0)
            
            cm_th = confusion_matrix(y_sub, pred_th, labels=[0, 1])
            tn, fp, fn, tp = cm_th.ravel() if cm_th.size==4 else (0,0,0,0)
            spec_th = tn/(tn+fp) if (tn+fp)>0 else 0
            npv_th = tn/(tn+fn) if (tn+fn)>0 else 0
            
            # 標記哪一個是原本的模型設定
            is_original = abs(th - original_threshold) < 1e-9
            marker = "⭐" if is_original else "  "
            
            print(f"      {marker} Th={th:.2f} | F1={f1_th:.4f} | Recall={rec_th:.4f} | Spec={spec_th:.4f} | Acc={acc_th:.4f}")
            
            # 如果是原本的閾值，暫存起來供後續儲存
            if is_original:
                final_stats = {
                    "pred": pred_th, "f1": f1_th, "acc": acc_th, 
                    "prec": prec_th, "rec": rec_th, "spec": spec_th, "npv": npv_th,
                    "threshold": th
                }
        
        # 確保有存到結果 (以防浮點數誤差，若沒對應到就用最後一個或原本的)
        if not final_stats:
             # Fallback: 使用原本閾值再算一次
             pred = (proba >= original_threshold).astype(int)
             cm = confusion_matrix(y_sub, pred, labels=[0, 1])
             tn, fp, fn, tp = cm.ravel()
             final_stats = {
                "pred": pred,
                "f1": f1_score(y_sub, pred),
                "acc": accuracy_score(y_sub, pred),
                "prec": precision_score(y_sub, pred, zero_division=0),
                "rec": recall_score(y_sub, pred, zero_division=0),
                "spec": tn/(tn+fp) if (tn+fp)>0 else 0,
                "npv": tn/(tn+fn) if (tn+fn)>0 else 0,
                "threshold": original_threshold
             }

        # 7. 儲存結果 (使用原本閾值的表現)
        results.append({
            "Label": label, 
            "F1": final_stats['f1'], "Acc": final_stats['acc'], "AUC": auc_val, 
            "Spec": final_stats['spec'], "NPV": final_stats['npv'], 
            "Prec": final_stats['prec'], "Recall": final_stats['rec'],
            "Threshold": final_stats['threshold']
        })
        
        # 8. 驗證集繪圖 (使用原本閾值的預測)
        viz = Visualizer(label, val_out_dir, sub_folder=label)
        viz.plot_confusion_matrix_aggregated(y_sub, final_stats['pred'])
        if len(np.unique(y_sub)) > 1:
            fpr, tpr, _ = roc_curve(y_sub, proba)
            viz.plot_roc_curve_with_ci([tpr], fpr, [auc_val])
            
    if results:
        res_df = pd.DataFrame(results)
        excel_path = os.path.join(val_out_dir, "External_Validation_Results.xlsx")
        res_df.to_excel(excel_path, index=False)
        
        cols = ["Label", "F1", "Prec", "Recall", "Spec", "NPV", "AUC", "Acc", "Threshold"]
        
        pretty_print_table(res_df[cols], title="外部驗證結果摘要 (Original Threshold)")
        print(f"\n外部驗證完成，結果已儲存至: {val_out_dir}")
    else:
        print("\n無法產生任何驗證結果 (可能缺資料或模型)")