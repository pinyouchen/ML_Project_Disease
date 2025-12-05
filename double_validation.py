import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load

# 引用您的模組
from processors import DataProcessor
from normative_modeling import NormativeModeler

# 設定繪圖風格
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft JhengHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def run_double_validation(train_run_dir, file_path, sheet_name, target_hrv='SDNN'):
    print("="*70)
    print(f"🚀 啟動雙重驗證 (Double Validation)")
    print(f"   模型來源: {train_run_dir}")
    print(f"   生理指標: {target_hrv}")
    print("="*70)

    # 1. 準備輸出資料夾
    out_dir = os.path.join(train_run_dir, "double_validation")
    os.makedirs(out_dir, exist_ok=True)

    # 2. 載入資料
    # 這裡我們需要所有資料來做分析
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # 初始化處理器 (為了做特徵工程)
    processor = DataProcessor(file_path, sheet_name, mode='all')
    processor.df = df
    processor.prepare_features_and_labels() # 產生 HRV_Mean, Ratio 等特徵
    
    # 3. 建立常模 (Normative Model) - 取得 Z-Score
    print(f"\n[1/2] 建立 {target_hrv} 常模...")
    norm_modeler = NormativeModeler(file_path, sheet_name, target=target_hrv, log_transform=True)
    if not norm_modeler.load_data(): return
    norm_modeler.train_health_model()
    norm_modeler.predict_deviations()
    
    # 取得 Z-Score 與 Traffic Light
    norm_modeler.apply_traffic_light_system()
    df_norm = norm_modeler.df[['Z_Score', 'Traffic_Light']].copy()
    
    # 4. 載入分類器 (Classifier) - 取得 預測機率
    print(f"\n[2/2] 載入分類模型與預測...")
    
    models_dir = os.path.join(train_run_dir, "models")
    disease_labels = ['SSD', 'MDD', 'Panic', 'GAD']
    
    for label in disease_labels:
        print(f"\n🔍 分析疾病: {label}")
        meta_path = os.path.join(models_dir, f"{label}_best.json")
        if not os.path.exists(meta_path):
            print(f"   ⚠️ 找不到 {label} 的模型，跳過。")
            continue
            
        with open(meta_path, 'r', encoding='utf-8') as f: meta = json.load(f)
        
        # 載入模型與轉換器
        model = load(os.path.join(models_dir, meta['files']['model']))
        scaler = load(os.path.join(models_dir, meta['files']['scaler'])) if meta['files']['scaler'] else None
        imputer = load(os.path.join(models_dir, meta['files']['imputer'])) if meta['files']['imputer'] else None
        
        # 準備 X (針對該模型需要的特徵)
        # 注意：這裡我們對"全體資料"做預測，看看分佈
        X_test = processor.apply_external_transform(
            processor.X, meta['feature_columns'], meta['outlier_bounds'], imputer, scaler
        )
        
        # 預測機率
        try:
            probs = model.predict_proba(X_test)[:, 1]
        except:
            print(f"   ⚠️ 模型不支援機率預測，跳過。")
            continue
            
        # --- 5. 整合數據 (Merge) ---
        # 我們只分析「臨床確診為疾病組」的人，看看 AI 跟 常模 怎麼說他們
        # (當然也可以看全體，但看疾病組最有意義)
        mask_disease = df[label] == 1
        
        analysis_df = pd.DataFrame({
            'Probability': probs[mask_disease],  # X軸: 分類器信心
            'Z_Score': df_norm.loc[mask_disease, 'Z_Score'], # Y軸: 生理偏差
            'Traffic_Light': df_norm.loc[mask_disease, 'Traffic_Light']
        })
        
        if len(analysis_df) == 0: continue

        # --- 6. 繪製雙重驗證圖 (2D Plot) ---
        plt.figure(figsize=(10, 8))
        
        # 畫象限分隔線
        # X軸切分點: 0.5 (或模型的最佳 threshold)
        th = meta.get('threshold', 0.5)
        plt.axvline(th, color='gray', linestyle='--', linewidth=1, label=f'Clf Threshold ({th:.2f})')
        
        # Y軸切分點: -1.96 (生理異常線)
        plt.axhline(-1.96, color='red', linestyle='--', linewidth=1, label='Physio Abnormal (-1.96)')
        
        # 散佈圖
        # 根據紅綠燈上色
        colors = {'Green': 'green', 'Yellow': 'orange', 'Red': 'red'}
        sns.scatterplot(data=analysis_df, x='Probability', y='Z_Score', 
                        hue='Traffic_Light', palette=colors, style='Traffic_Light', 
                        s=80, alpha=0.7)
        
        # 標註象限意義
        # 右下 (High Prob, Low Z): 雙重確診
        plt.text(0.95, -3, "Double Confirmed\n(Physio+)", ha='right', va='bottom', fontsize=12, color='darkred', fontweight='bold')
        # 右上 (High Prob, Normal Z): 心理/認知型
        plt.text(0.95, 1, "Psychological Type\n(Physio-)", ha='right', va='top', fontsize=12, color='darkblue')
        # 左下 (Low Prob, Low Z): 漏網之魚
        plt.text(0.05, -3, "Missed Risk\n(Physio+)", ha='left', va='bottom', fontsize=12, color='darkorange', fontweight='bold')
        
        plt.title(f'Double Validation: {label} vs {target_hrv}', fontsize=15)
        plt.xlabel(f'Classifier Probability (Model Confidence)')
        plt.ylabel(f'{target_hrv} Z-Score (Physiological Status)')
        plt.xlim(0, 1)
        plt.ylim(-4, 4)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.tight_layout()
        
        save_path = os.path.join(out_dir, f"DoubleValid_{label}_{target_hrv}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"   📊 圖表已儲存: {save_path}")
        
        # 計算「漏網之魚」比例 (AI 判 < Th，但 Z < -1.96)
        missed = analysis_df[(analysis_df['Probability'] < th) & (analysis_df['Z_Score'] < -1.96)]
        print(f"   🔍 發現 {len(missed)} 位潛在生理異常患者被分類器漏判 (Low Probability)")

    print(f"\n✅ 雙重驗證完成！請查看: {out_dir}")

# ==========================================
# 執行區
# ==========================================
if __name__ == "__main__":
    # 1. 設定檔案路徑
    FILE_PATH = r"D:\FLY114-main\data.xlsx"
    SHEET_NAME = "Data2"
    
    # 2. 設定訓練好的模型資料夾 (請改成您實際跑出來的資料夾名稱)
    # 例如: runs/Run_all_20251130_120000
    # 請務必確認這個資料夾裡有 models 子資料夾
    TRAIN_RUN_DIR = r"D:\ML_Project\runs\Task5_Full_D2_20251127_151301"  
    
    # 3. 選擇一個最具代表性的生理指標 (通常是 SDNN 或 HF)
    TARGET_HRV = 'SDNN'
    
    if os.path.exists(TRAIN_RUN_DIR):
        run_double_validation(TRAIN_RUN_DIR, FILE_PATH, SHEET_NAME, target_hrv=TARGET_HRV)
    else:
        print(f"❌ 找不到模型資料夾: {TRAIN_RUN_DIR}，請修改程式碼中的路徑。")