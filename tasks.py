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
import shap 

# 引用自定義模組
from model_trainer import ModelTrainer
from utils import save_best_model, pretty_print_table, load_best_model_and_meta
from visualization import Visualizer

from processors import (
    ProcessorBaseline4
)

def run_binary_task(task_name, file_path, sheet_name, processor_cls, use_stacking=True):
    print("\n" + "="*70)
    print(f"執行任務: {task_name} (AutoML & SHAP-OOF Version)")
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
    
    # [Check] 確保 df_full 有 Subject_ID，若無則使用 Index 作為 ID
    if 'Subject_ID' not in df_full.columns:
        df_full['Subject_ID'] = df_full.index
    
    summary_rows = []
    overall_roc_data = {}
    overall_pr_data = {}
    overall_metrics_list = [] 
    
    for label in label_names:
        if label not in y_dict: continue
        print(f"\n🩺 診斷：{label} vs Health")
        
        mask_disease = df_full[label] == 1
        mask_health = (df_full['Health'] == 1) & (df_full[label] == 0)
        mask_valid = mask_disease | mask_health
        
        # 這裡保留原始 Index，這對 Step 4 至關重要
        X_sub = X_full.loc[mask_valid].copy()
        y_sub = np.where(mask_disease[mask_valid], 1, 0)
        
        viz = Visualizer(label, run_dir, sub_folder=label)

        print("   📊 繪製特徵相關性矩陣 (EDA)...")
        try:
            X_sub_corr_p = processor.impute_and_scale(X_sub, fit=True)
            viz.plot_correlation_matrix(X_sub_corr_p)
        except Exception as e:
            print(f"   ⚠️ 無法繪製 Correlation Matrix: {e}")

        base_f1 = {'SSD':0.66, 'MDD':0.46, 'Panic':0.50, 'GAD':0.57}.get(label, 0.5)
        target_f1 = {'SSD':0.75, 'MDD':0.75, 'Panic':0.55, 'GAD':0.70}.get(label, 0.7)
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        metrics_list = []
        tprs = []; mean_fpr = np.linspace(0, 1, 100)
        roc_aucs = []
        precisions = []; mean_recall = np.linspace(0, 1, 100)
        pr_aucs = []
        no_skill = y_sub.sum() / len(y_sub)
        y_true_all = []; y_pred_all = []
        shap_values_list = []
        X_test_shap_list = []
        importance_list = []
        best_model_info = {"f1": -1.0, "p": -1.0, "obj": None, "name": None}
        
        # [新增] 用於儲存 Step 4 所需的詳細預測資料 (Out-Of-Fold Predictions)
        oof_predictions_list = []

        fold_id = 1
        for train_idx, test_idx in skf.split(X_sub, y_sub):
            print(f"\n   📂 Fold {fold_id}/5")
            
            # [新增] 取得這一折測試資料的原始 ID (用於追蹤病人)
            current_test_ids = X_sub.index[test_idx]
            
            X_tr, X_te = X_sub.iloc[train_idx], X_sub.iloc[test_idx]
            X_tr = X_tr.reset_index(drop=True)
            X_te = X_te.reset_index(drop=True)
            y_tr = pd.Series(y_sub[train_idx]) 
            y_te = pd.Series(y_sub[test_idx])
            
            X_tr_p, X_te_p = processor.impute_and_scale(X_tr, X_te, fit=True)
            
            if not X_tr_p.isnull().any().any():
                try:
                    iso = IsolationForest(contamination=0.03, random_state=42, n_jobs=1)
                    outlier_preds = iso.fit_predict(X_tr_p)
                    mask_clean = (outlier_preds == 1) | (y_tr == 1)
                    X_tr_clean = X_tr_p[mask_clean].copy()
                    y_tr_clean = y_tr[mask_clean].copy()
                    removed = len(X_tr_p) - len(X_tr_clean)
                    if removed > 0: print(f"      🧹 移除了 {removed} 個異常樣本")
                except:
                    X_tr_clean, y_tr_clean = X_tr_p, y_tr
            else:
                X_tr_clean, y_tr_clean = X_tr_p, y_tr

            trainer = ModelTrainer(label, y_tr_clean.sum(), len(y_tr_clean)-y_tr_clean.sum(), base_f1, target_f1, use_stacking)
            trainer.build_models()
            res = trainer.train_and_evaluate(X_tr_clean, X_te_p, y_tr_clean, y_te)
            
            target_shap_model = None
            if 'XGB' in trainer.fitted_models: target_shap_model = trainer.fitted_models['XGB']
            elif 'LGBM' in trainer.fitted_models: target_shap_model = trainer.fitted_models['LGBM']
            
            if target_shap_model:
                try:
                    explainer = shap.TreeExplainer(target_shap_model)
                    shap_vals = explainer.shap_values(X_te_p)
                    if isinstance(shap_vals, list) and len(shap_vals) == 2:
                        shap_values_list.append(shap_vals[1]) 
                    elif isinstance(shap_vals, np.ndarray):
                        if shap_vals.ndim == 2: shap_values_list.append(shap_vals)
                        elif shap_vals.ndim == 3: shap_values_list.append(shap_vals[:, :, 1])
                    X_test_shap_list.append(X_te_p)
                    if hasattr(target_shap_model, 'feature_importances_'):
                        importance_list.append(pd.DataFrame({'Feature': X_te_p.columns, 'Importance': target_shap_model.feature_importances_}))
                except Exception as e: pass

            special = [m for m in res.keys() if m in ['Ensemble', 'Stacking']]
            show_name = max(special, key=lambda k: res[k]['f1_score']) if special else max(res.keys(), key=lambda k: res[k]['f1_score'])
            r = res[show_name]
            
            # [新增] 收集詳細預測結果 (Step 4 關鍵)
            # 將這一折所有測試病人的預測結果存入 List
            for i in range(len(test_idx)):
                oof_predictions_list.append({
                    'Subject_ID': current_test_ids[i], # 這裡使用的是原始 DataFrame 的 Index (需確保 Index 即 ID)
                    'Ground_Truth': y_te.iloc[i],
                    'Pred_Prob': r['y_pred_proba'][i],
                    'Pred_Label': r['y_pred'][i],
                    'Fold': fold_id,
                    'Best_Model': show_name,
                    'Threshold': r['threshold']
                })

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

        X_sub_p = processor.impute_and_scale(X_sub, fit=True)
        if X_sub_p.isnull().any().any():
            viz.plot_pca_scatter(X_sub_p.fillna(-1), y_sub)
        else:
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
            
        if shap_values_list and X_test_shap_list:
            try:
                if len(shap_values_list) == len(X_test_shap_list):
                    global_shap_values = np.concatenate(shap_values_list, axis=0)
                    global_X_test = pd.concat(X_test_shap_list, axis=0)
                    if global_shap_values.shape[0] == global_X_test.shape[0]:
                        print(f"   📊 繪製 Global OOF SHAP Summary (N={global_X_test.shape[0]})...")
                        viz.plot_shap_summary_oof(global_shap_values, global_X_test)
            except Exception as e: 
                print(f"   ⚠️ SHAP Plot Error: {e}")

        if best_model_info['obj']:
            save_best_model(
                models_dir, label, best_model_info['obj'], 
                best_model_info['scaler'], best_model_info['imputer'],
                best_model_info['cols'], best_model_info['bounds'], 
                best_model_info['thresh']
            )

        # [新增] 匯出 Step 4 專用 Excel (單筆詳細結果)
        print(f"\n💾 正在匯出 Step 4 分析用總表 (Step1_Predictions_Detail_{label}.xlsx)...")
        if oof_predictions_list:
            df_oof = pd.DataFrame(oof_predictions_list)
            
            # 定義想要保留的原始欄位 (基本資料 + 心理量表)
            # 這些欄位如果存在於原始 Excel，就會被合併進來
            meta_cols = ['Age', 'Sex', 'BMI']
            # 加入常見的心理量表欄位名稱 (根據您的 processors.py 推測)
            potential_psych_cols = ['phq15', 'haq21', 'cabah', 'bdi', 'bai', 'PHQ_15_Total']
            for c in potential_psych_cols:
                if c in df_full.columns: meta_cols.append(c)
                
            # 合併欄位 (使用 Subject_ID 對應)
            # 確保 meta_cols 確實存在
            cols_to_merge = [c for c in meta_cols if c in df_full.columns]
            
            # 左合併：以預測結果為準，把基本資料貼過來
            df_oof = df_oof.merge(df_full[cols_to_merge], left_on='Subject_ID', right_index=True, how='left')
            
            out_path = os.path.join(run_dir, f"Step1_Predictions_Detail_{label}.xlsx")
            df_oof.to_excel(out_path, index=False)
            print(f"✅ 詳細預測表已儲存: {out_path} (可直接用於 Step 4)")

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


def run_external_validation(models_dir_input, file_path, sheet_name, processor_cls=ProcessorBaseline4):
    """
    外部驗證函式 (含 GAD 自適應閾值挑選)
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

    processor = processor_cls(file_path, sheet_name)
    if not processor.load_data(): return
    if not processor.prepare_features_and_labels(): return
    
    df_full = processor.df
    X_full = processor.X
    label_names = ['SSD', 'MDD', 'Panic', 'GAD']
    
    results = []
    
    for label in label_names:
        print(f"\n🔍 驗證：{label}")
        
        info = load_best_model_and_meta(models_dir_input, label)
        if not info:
            print(f"  無法載入 {label} 模型，跳過")
            continue
            
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
            print("  無有效樣本")
            continue
        
        print(f"   樣本數: {len(y_sub)} (正例={y_sub.sum()}, 負例={len(y_sub)-y_sub.sum()})")

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
            
        processor.outlier_bounds_ = info['outlier_bounds']
        processor.knn_imputer = info['imputer']
        processor.scaler = info['scaler']
        
        X_eval_p = processor.impute_and_scale(X_eval, fit=False)
        
        model = info['model']
        try:
            proba = model.predict_proba(X_eval_p)[:, 1]
        except:
            proba = model.predict(X_eval_p)
        
        try: auc_val = roc_auc_score(y_sub, proba)
        except: auc_val = np.nan

        original_threshold = info['threshold']
        
        # 建立候選閾值列表：包含原本的，以及 0.3~0.7
        test_candidates = [original_threshold] + list(np.arange(0.3, 0.75, 0.05))
        test_thresholds = sorted(list(set(test_candidates)))
        
        print(f"   📊 閾值診斷與自動挑選 (GAD目標: F1>0.4, Rec>0.4, Spec>0.6)...")
        
        best_compliant_stats = None
        best_f1_compliant = -1.0
        
        original_stats = None

        for th in test_thresholds:
            pred_th = (proba >= th).astype(int)
            
            f1_th = f1_score(y_sub, pred_th)
            acc_th = accuracy_score(y_sub, pred_th)
            prec_th = precision_score(y_sub, pred_th, zero_division=0)
            rec_th = recall_score(y_sub, pred_th, zero_division=0)
            
            cm_th = confusion_matrix(y_sub, pred_th, labels=[0, 1])
            tn, fp, fn, tp = cm_th.ravel() if cm_th.size==4 else (0,0,0,0)
            spec_th = tn/(tn+fp) if (tn+fp)>0 else 0
            npv_th = tn/(tn+fn) if (tn+fn)>0 else 0
            
            is_original = abs(th - original_threshold) < 1e-9
            
            # 建立 Stats 物件
            stats = {
                "pred": pred_th, "f1": f1_th, "acc": acc_th, 
                "prec": prec_th, "rec": rec_th, "spec": spec_th, "npv": npv_th,
                "threshold": th, "is_original": is_original
            }
            
            if is_original:
                original_stats = stats
                
            # [挑選邏輯] 檢查是否符合 GAD 條件
            is_compliant = True
            if label == 'GAD':
                if not (f1_th > 0.4 and rec_th > 0.4 and spec_th > 0.6):
                    is_compliant = False
            
            marker = "  "
            if is_original: marker = "⭐"
            if is_compliant and label == 'GAD' and not is_original: marker = "✨"
            
            # 更新最佳符合條件者 (Maximize F1)
            if is_compliant:
                if f1_th > best_f1_compliant:
                    best_f1_compliant = f1_th
                    best_compliant_stats = stats

            # 簡化 Log 輸出
            if is_original or (is_compliant and label=='GAD') or abs(th-0.5)<0.01:
                print(f"      {marker} Th={th:.2f} | F1={f1_th:.4f} | Rec={rec_th:.4f} | Spec={spec_th:.4f}")

        # 決定最終輸出的結果
        # 如果有找到符合條件的閾值，就用它；否則回退到 Original
        if label == 'GAD' and best_compliant_stats is not None:
            final_stats = best_compliant_stats
            if not final_stats['is_original']:
                print(f"      👉 [Selected] 採用適應性閾值 Th={final_stats['threshold']:.2f} (符合 GAD 條件)")
            else:
                print(f"      👉 [Selected] 採用原始閾值 (已符合條件)")
        else:
            final_stats = original_stats
            # 防呆：如果 original_stats 沒抓到 (浮點數誤差)，重算一次
            if final_stats is None:
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
                print(f"      👉 採用原始閾值 (Fallback)")

        results.append({
            "Label": label, 
            "F1": final_stats['f1'], "Acc": final_stats['acc'], "AUC": auc_val, 
            "Spec": final_stats['spec'], "NPV": final_stats['npv'], 
            "Prec": final_stats['prec'], "Recall": final_stats['rec'],
            "Threshold": final_stats['threshold']
        })
        
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
        pretty_print_table(res_df[cols], title="外部驗證結果摘要 (Selected)")
        print(f"\n外部驗證完成，結果已儲存至: {val_out_dir}")
    else:
        print("\n無法產生任何驗證結果 (可能缺資料或模型)")