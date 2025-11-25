import os
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, roc_auc_score
from sklearn.ensemble import IsolationForest

from model_trainer import ModelTrainer
from utils import save_best_model, pretty_print_table, load_best_model_and_meta
from visualization import Visualizer
import shap 

# 🔥 修正：這裡補上了 DataProcessorBaseline
from processors import (
    ProcessorHRV, 
    ProcessorPsych, 
    ProcessorBaselineAll, 
    ProcessorFullV62, 
    DataProcessorBaseline
)

def run_binary_task(task_name, file_path, sheet_name, processor_cls, use_stacking=True):
    print("\n" + "="*70)
    print(f"🚀 執行任務: {task_name} (V6.12 完美還原版)")
    print("="*70)
    
    timestamp = datetime.now().strftime(f"{task_name}_%Y%m%d_%H%M%S")
    run_dir = os.path.join(os.getcwd(), "runs", timestamp)
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    
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
        
        mask_disease = df_full[label] == 1
        mask_health = (df_full['Health'] == 1) & (df_full[label] == 0)
        mask_valid = mask_disease | mask_health
        X_sub = X_full.loc[mask_valid].copy()
        y_sub = np.where(mask_disease[mask_valid], 1, 0)
        
        base_f1 = {'SSD':0.66, 'MDD':0.46, 'Panic':0.50, 'GAD':0.57}.get(label, 0.5)
        target_f1 = {'SSD':0.75, 'MDD':0.75, 'Panic':0.55, 'GAD':0.70}.get(label, 0.7)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        metrics_list = []
        tprs = []; mean_fpr = np.linspace(0, 1, 100)
        roc_aucs = []
        precisions = []; mean_recall = np.linspace(0, 1, 100)
        pr_aucs = []
        no_skill = y_sub.sum() / len(y_sub)
        
        y_true_all = []
        y_pred_all = []
        
        shap_values_list = []
        X_test_shap_list = []
        importance_list = []
        
        # 初始化最佳模型記錄
        best_model_info = {"f1": -1.0, "p": -1.0, "obj": None, "name": None}
        
        fold_id = 1
        for train_idx, test_idx in skf.split(X_sub, y_sub):
            print(f"\n   📂 Fold {fold_id}/5")
            X_tr, X_te = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
            
            # 1. 索引重置
            X_tr = X_tr.reset_index(drop=True)
            X_te = X_te.reset_index(drop=True)
            y_tr = pd.Series(y_sub[train_idx]) 
            y_te = pd.Series(y_sub[test_idx])
            
            # 2. 前處理
            X_tr_p, X_te_p = processor.impute_and_scale(X_tr, X_te, fit=True)
            
            # 3. Isolation Forest (固定 n_jobs=1)
            iso = IsolationForest(contamination=0.03, random_state=42, n_jobs=1)
            outlier_preds = iso.fit_predict(X_tr_p)
            mask_clean = (outlier_preds == 1) | (y_tr == 1)
            X_tr_clean = X_tr_p[mask_clean].copy()
            y_tr_clean = y_tr[mask_clean].copy()
            
            removed_count = len(X_tr_p) - len(X_tr_clean)
            if removed_count > 0:
                print(f"      🧹 IsolationForest 移除了 {removed_count} 個異常樣本")

            # 4. 訓練
            trainer = ModelTrainer(label, y_tr_clean.sum(), len(y_tr_clean)-y_tr_clean.sum(), base_f1, target_f1, use_stacking)
            trainer.build_models()
            res = trainer.train_and_evaluate(X_tr_clean, X_te_p, y_tr_clean, y_te)
            
            # 5. SHAP 收集 (強制指定樹模型)
            target_shap_model = None
            if 'XGB' in trainer.fitted_models: target_shap_model = trainer.fitted_models['XGB']
            elif 'LGBM' in trainer.fitted_models: target_shap_model = trainer.fitted_models['LGBM']
            
            if target_shap_model:
                try:
                    explainer = shap.TreeExplainer(target_shap_model)
                    shap_vals = explainer.shap_values(X_te_p)
                    if isinstance(shap_vals, list): shap_values_list.append(shap_vals[1]) 
                    else: shap_values_list.append(shap_vals)
                    X_test_shap_list.append(X_te_p)
                    if hasattr(target_shap_model, 'feature_importances_'):
                        importance_list.append(pd.DataFrame({
                            'Feature': X_te_p.columns, 'Importance': target_shap_model.feature_importances_
                        }))
                except Exception as e: print(f"      ⚠️ SHAP 計算失敗: {e}")

            # 6. 選擇當折最佳模型 (用於 Metrics 統計)
            special = [m for m in res.keys() if m in ['Ensemble', 'Stacking']]
            if special:
                show_name = max(special, key=lambda k: res[k]['f1_score'])
            else:
                show_name = max(res.keys(), key=lambda k: res[k]['f1_score'])
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
            
            # 7. 更新整體最佳單一模型 (用於保存)
            singles = [k for k in res.keys() if k not in ['Ensemble', 'Stacking']]
            if singles:
                best_s_name = max(singles, key=lambda k: res[k]['f1_score'])
                best_s = res[best_s_name]
                
                # 若 F1 相同，再比 Precision (Tie-Breaker 還原)
                is_better = False
                if best_s['f1_score'] > best_model_info['f1']:
                    is_better = True
                elif best_s['f1_score'] == best_model_info['f1']:
                    if best_s['precision'] > best_model_info['p']:
                        is_better = True
                
                if is_better:
                    best_model_info = {
                        "f1": best_s['f1_score'], "p": best_s['precision'],
                        "obj": best_s['model'], "name": best_s_name, "fold": fold_id,
                        "thresh": best_s['threshold'], "scaler": processor.scaler,
                        "imputer": processor.knn_imputer, "cols": X_tr_p.columns,
                        "bounds": processor.outlier_bounds_,
                        "r": best_s['recall'], "spec": best_s['specificity'], "npv": best_s['npv'],
                        "auc": best_s['auc'], "acc": best_s['accuracy']
                    }
            fold_id += 1

        # 8. 繪圖與儲存
        viz = Visualizer(label, run_dir, sub_folder=label)
        
        # 繪製 PCA 散佈圖
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
        
        if importance_list: viz.plot_feature_importance_boxplot(pd.concat(importance_list))
        if shap_values_list:
            try:
                viz.plot_shap_summary_oof(np.concatenate(shap_values_list, axis=0), pd.concat(X_test_shap_list, axis=0))
            except Exception as e: print(f"SHAP Error: {e}")

        if best_model_info['obj']:
            save_best_model(models_dir, label, best_model_info['obj'], 
                            best_model_info['scaler'], best_model_info['imputer'],
                            best_model_info['cols'], best_model_info['bounds'], best_model_info['thresh'])

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
                "F1(Best)", "P(Best)", "R(Best)", "Spec(Best)", "AUC(Best)", "ACC(Best)", 
                "F1(avg)", "P(avg)", "R(avg)", "Spec(avg)", "AUC(avg)", "ACC(avg)"]
        pretty_print_table(res_df[cols], title="最終結果摘要")
        print(f"\n✅ 所有結果與比較圖表已存至: {run_dir}/plots")

# ==========================================
# 外部驗證函式 (External Validation)
# ==========================================
def run_external_validation(models_dir_input, file_path, sheet_name):
    print("\n" + "="*70)
    print(f"🚀 執行外部驗證 (Data1)")
    print(f"📂 模型來源: {models_dir_input}")
    print("="*70)
    
    if not os.path.exists(models_dir_input):
        print(f"❌ 找不到模型資料夾: {models_dir_input}")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    val_out_dir = os.path.join(os.path.dirname(models_dir_input), f"Validation_Data1_{timestamp}")
    os.makedirs(val_out_dir, exist_ok=True)

    # 預設使用 DataProcessorBaseline (對應 baseline 任務)
    # 若要支援 FullV62，這裡可以改成判斷或傳參數
    processor = DataProcessorBaseline(file_path, sheet_name)
    
    if not processor.load_data(): return
    if not processor.prepare_features_and_labels(): return
    
    df_full = processor.df
    X_full = processor.X
    label_names = ['SSD', 'MDD', 'Panic', 'GAD']
    
    results = []
    
    for label in label_names:
        print(f"\n🔍 驗證：{label}")
        
        # 載入模型
        info = load_best_model_and_meta(models_dir_input, label)
        if not info:
            print(f"   ⚠️ 無法載入 {label} 模型，跳過")
            continue
            
        # 篩選資料
        if label not in df_full.columns or 'Health' not in df_full.columns: continue
        
        mask_valid = (
            df_full['Health'].isin([0, 1]) &
            df_full[label].isin([0, 1]) &
            ((df_full['Health'] == 1) | (df_full[label] == 1))
        )
        df_sub = df_full.loc[mask_valid].copy()
        X_sub = X_full.loc[mask_valid].copy()
        
        mask_xor = (df_sub['Health'] == 1) ^ (df_sub[label] == 1)
        df_sub = df_sub.loc[mask_xor]
        X_sub = X_sub.loc[mask_xor]
        
        y_sub = np.where(df_sub[label] == 1, 1, 0)
        
        if len(y_sub) == 0:
            print("   ⚠️ 無有效樣本")
            continue
        
        print(f"   樣本數: {len(y_sub)} (正例={y_sub.sum()}, 負例={len(y_sub)-y_sub.sum()})")

        # 對齊特徵
        required_cols = info['feature_columns']
        X_eval = pd.DataFrame(index=X_sub.index)
        for col in required_cols:
            if col in X_sub.columns:
                X_eval[col] = X_sub[col]
            else:
                X_eval[col] = np.nan
        X_eval = X_eval[required_cols]
            
        # 套用訓練時的 Preprocessor
        processor.outlier_bounds_ = info['outlier_bounds']
        processor.knn_imputer = info['imputer']
        processor.scaler = info['scaler']
        
        X_eval_p = processor.impute_and_scale(X_eval, fit=False)
        
        # 推論
        model = info['model']
        try:
            proba = model.predict_proba(X_eval_p)[:, 1]
        except:
            proba = model.predict(X_eval_p)
            
        pred = (proba >= info['threshold']).astype(int)
        
        f1 = f1_score(y_sub, pred)
        acc = accuracy_score(y_sub, pred)
        prec = precision_score(y_sub, pred, zero_division=0)
        rec = recall_score(y_sub, pred, zero_division=0)
        
        cm = confusion_matrix(y_sub, pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        npv = tn/(tn+fn) if (tn+fn)>0 else 0
        
        try: auc_val = roc_auc_score(y_sub, proba)
        except: auc_val = np.nan
        
        print(f"   → Result: F1={f1:.4f}, Acc={acc:.4f}, AUC={auc_val:.4f}, Spec={spec:.4f}")
        
        results.append({
            "Label": label, "F1": f1, "Acc": acc, "AUC": auc_val, 
            "Spec": spec, "NPV": npv, "Prec": prec, "Recall": rec,
            "Threshold": info['threshold']
        })
        
        # 繪圖
        viz = Visualizer(label, val_out_dir, sub_folder=label)
        viz.plot_confusion_matrix_aggregated(y_sub, pred)
        if len(np.unique(y_sub)) > 1:
            fpr, tpr, _ = roc_curve(y_sub, proba)
            viz.plot_roc_curve_with_ci([tpr], fpr, [auc_val])
            
    if results:
        res_df = pd.DataFrame(results)
        excel_path = os.path.join(val_out_dir, "External_Validation_Results.xlsx")
        res_df.to_excel(excel_path, index=False)
        
        cols = ["Label", "F1", "Prec", "Recall", "Spec", "NPV", "AUC", "Acc"]
        
        pretty_print_table(res_df[cols], title="外部驗證結果摘要 (Data1)")
        print(f"\n✅ 外部驗證完成，結果已儲存至: {val_out_dir}")
    else:
        print("\n⚠️ 無法產生任何驗證結果 (可能缺資料或模型)")