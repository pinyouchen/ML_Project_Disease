# normative_modeling.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor

# 設定繪圖風格
sns.set_theme(style="whitegrid", context="talk")
plt.rcParams['font.sans-serif'] = ['Arial', 'Microsoft JhengHei', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class NormativeModeler:
    def __init__(self, file_path, sheet_name='Data2', 
                 features=['Age', 'Sex', 'BMI'], 
                 target='SDNN',
                 log_transform=True):
        """
        初始化常模建模器
        :param features: 協變量 (Age, Sex, BMI)
        :param target: 目標生理指標 (HRV features)
        :param log_transform: 是否對目標變數取 log1p (HRV 數據通常建議 True)
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.features = features
        self.target = target
        self.log_transform = log_transform
        
        self.df = None
        self.models = {}
        
    def load_data(self):
        """載入資料並進行基礎清洗"""
        if not os.path.exists(self.file_path):
            print(f"❌ 找不到檔案: {self.file_path}")
            return False
        
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            
            # 確保必要欄位存在
            cols_needed = self.features + [self.target, 'Health']
            if not all(col in self.df.columns for col in cols_needed):
                print(f"❌ 資料表缺少必要欄位。需要: {cols_needed}")
                return False
            
            # 移除空值
            self.df = self.df.dropna(subset=cols_needed)
            
            # [特徵工程] Log 轉換
            if self.log_transform:
                if (self.df[self.target] < 0).any():
                    print(f"⚠️ 警告: {self.target} 含有負值，跳過 Log 轉換")
                else:
                    self.df[f'{self.target}_Raw'] = self.df[self.target] # 備份原始值
                    self.df[self.target] = np.log1p(self.df[self.target])
            
            return True
        except Exception as e:
            print(f"❌ 載入失敗: {e}")
            return False

    def train_health_model(self):
        """
        [關鍵] 僅使用「健康組 (Health=1)」訓練模型
        """
        # 1. 篩選健康組
        mask_health = self.df['Health'] == 1
        X_health = self.df.loc[mask_health, self.features]
        y_health = self.df.loc[mask_health, self.target]
        
        # 2. 訓練 5%, 50%, 95% 分位數模型
        quantiles = [0.05, 0.50, 0.95]
        quantile_names = ['lower', 'median', 'upper']
        
        for q, name in zip(quantiles, quantile_names):
            model = LGBMRegressor(
                objective='quantile',
                alpha=q,
                n_estimators=100,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
            model.fit(X_health, y_health)
            self.models[name] = model

    def predict_deviations(self):
        """計算所有人的偏差分數"""
        X_all = self.df[self.features]
        
        self.df['Pred_Low'] = self.models['lower'].predict(X_all)
        self.df['Pred_Mid'] = self.models['median'].predict(X_all)
        self.df['Pred_High'] = self.models['upper'].predict(X_all)
        
        # 計算 W-score (類似 Z-score)
        norm_range = (self.df['Pred_High'] - self.df['Pred_Low'])
        norm_range = norm_range.replace(0, 1e-6)
        estimated_std = norm_range / 3.92 # 95% CI width -> std
        
        self.df['Z_Score'] = (self.df[self.target] - self.df['Pred_Mid']) / estimated_std
        self.df['Is_Abnormal_Low'] = self.df['Z_Score'] < -1.96

    def analyze_disease_groups(self, disease_labels=['SSD', 'MDD', 'Panic', 'GAD']):
        """
        統計異常率並回傳結果 (List of Dicts) 供 Excel 輸出使用
        """
        print(f"\n📊 分析報告: {self.target}")
        print(f"{'Group':<10} | {'Mean Z':<8} | {'Abnormal%':<10}")
        print("-" * 35)
        
        stats_results = [] # 用來儲存統計結果

        # 1. 健康組數據
        health_df = self.df[self.df['Health'] == 1]
        if len(health_df) > 0:
            z_mean = health_df['Z_Score'].mean()
            abn_rate = health_df['Is_Abnormal_Low'].mean()
            print(f"{'Healthy':<10} | {z_mean:>6.2f}   | {abn_rate:>6.1%}")
            
            stats_results.append({
                'Target': self.target,
                'Group': 'Healthy',
                'N': len(health_df),
                'Mean_Z_Score': z_mean,
                'Abnormal_Rate': abn_rate,
                'Is_High_Risk': False
            })
        
        # 2. 疾病組數據
        for label in disease_labels:
            if label in self.df.columns:
                sub_df = self.df[self.df[label] == 1]
                if len(sub_df) > 0:
                    z_mean = sub_df['Z_Score'].mean()
                    abn_rate = sub_df['Is_Abnormal_Low'].mean()
                    is_high_risk = abn_rate > 0.1
                    
                    flag = "🔴" if is_high_risk else "  "
                    print(f"{label:<10} | {z_mean:>6.2f}   | {abn_rate:>6.1%} {flag}")
                    
                    stats_results.append({
                        'Target': self.target,
                        'Group': label,
                        'N': len(sub_df),
                        'Mean_Z_Score': z_mean,
                        'Abnormal_Rate': abn_rate,
                        'Is_High_Risk': is_high_risk
                    })
        print("-" * 35)
        return stats_results

    def plot_normative_curves(self, save_dir=None):
        """
        繪製兩種圖表，並自動依 Target 建立子資料夾
        保留完整的繪圖邏輯 (分性別 + Z-Score)
        """
        target_dir = None
        if save_dir:
            # 為每個 Target 建立專屬子資料夾
            target_dir = os.path.join(save_dir, self.target)
            os.makedirs(target_dir, exist_ok=True)

        # --- 圖表 1: 分性別的原始常模圖 (Raw Value vs Age) ---
        sex_map = {1: 'Male', 0: 'Female'}
        
        for sex_val, sex_name in sex_map.items():
            plt.figure(figsize=(10, 6))
            
            # 1. 產生該性別的標準曲線 (固定 BMI=24)
            age_range = np.linspace(self.df['Age'].min(), self.df['Age'].max(), 100)
            X_dummy = pd.DataFrame({
                'Age': age_range,
                'Sex': [sex_val] * 100,
                'BMI': [24] * 100
            })
            
            y_low = self.models['lower'].predict(X_dummy)
            y_mid = self.models['median'].predict(X_dummy)
            y_high = self.models['upper'].predict(X_dummy)
            
            # 畫背景帶
            plt.fill_between(age_range, y_low, y_high, color='green', alpha=0.1, label=f'Healthy Range ({sex_name})')
            plt.plot(age_range, y_mid, color='green', linestyle='--', label='Median Trend')
            
            # 2. 畫該性別的真實數據點
            df_sex = self.df[self.df['Sex'] == sex_val]
            
            # 健康組
            healthy = df_sex[df_sex['Health'] == 1]
            plt.scatter(healthy['Age'], healthy[self.target], c='gray', s=20, alpha=0.3, label='Healthy')
            
            # 異常病人
            abnormal = df_sex[(df_sex['Health'] == 0) & (df_sex['Is_Abnormal_Low'])]
            plt.scatter(abnormal['Age'], abnormal[self.target], c='red', marker='x', s=60, alpha=0.9, label='Abnormal Pts')
            
            # 正常病人
            normal_pt = df_sex[(df_sex['Health'] == 0) & (~df_sex['Is_Abnormal_Low'])]
            plt.scatter(normal_pt['Age'], normal_pt[self.target], c='blue', s=20, alpha=0.4, label='Normal Pts')
            
            plt.title(f'{self.target} Normative Curve ({sex_name})', fontsize=14)
            plt.xlabel('Age')
            plt.ylabel(f'{self.target} Value')
            plt.legend()
            plt.tight_layout()
            
            if target_dir:
                plt.savefig(os.path.join(target_dir, f"Normative_Raw_{sex_name}.png"), dpi=300)
                plt.close()
            else:
                plt.show()

        # --- 圖表 2: Z-Score 偏差圖 ---
        plt.figure(figsize=(12, 7))
        
        plt.axhline(0, color='green', linestyle='--', linewidth=1.5, label='Healthy Median')
        plt.axhline(-1.96, color='red', linestyle='--', linewidth=1.5, label='Lower Limit (95%)')
        plt.axhspan(-1.96, 1.96, color='green', alpha=0.05, label='Normal Range')
        
        disease_df = self.df[self.df['Health'] == 0].copy()
        
        def get_disease_label(row):
            if row.get('Panic', 0) == 1: return 'Panic'
            if row.get('MDD', 0) == 1: return 'MDD'
            if row.get('GAD', 0) == 1: return 'GAD'
            if row.get('SSD', 0) == 1: return 'SSD'
            return 'Other'

        disease_df['Disease_Type'] = disease_df.apply(get_disease_label, axis=1)
        
        sns.scatterplot(data=disease_df, x='Age', y='Z_Score', hue='Disease_Type', 
                        style='Is_Abnormal_Low', palette='deep', s=60, alpha=0.8)
        
        plt.title(f'{self.target} Deviation Map (Z-Score)', fontsize=16)
        plt.ylabel('Deviation from Norm (Z-Score)')
        plt.xlabel('Age')
        plt.ylim(-5, 5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if target_dir:
            plt.savefig(os.path.join(target_dir, f"Normative_ZScore.png"), dpi=300)
            plt.close()
            print(f"📊 圖表已儲存至資料夾: {target_dir}")
        else:
            plt.show()

# ==========================================
# 主執行區
# ==========================================
if __name__ == "__main__":
    # 請修改為您的檔案路徑
    FILE_PATH = r"D:\FLY114-main\data.xlsx"
    SHEET_NAME = "Data2"
    
    hrv_targets = ['MEANH', 'SDNN', 'TP', 'VLF', 'LF', 'HF', 'NLF', 'LFHF']
    out_dir = os.path.join(os.getcwd(), "runs", "normative_analysis")
    
    # 用來收集所有指標的統計結果
    all_targets_summary = []
    
    print(f"🚀 啟動常模建模 (共 {len(hrv_targets)} 個指標)")
    
    for target in hrv_targets:
        modeler = NormativeModeler(FILE_PATH, SHEET_NAME, target=target, log_transform=True)
        if modeler.load_data():
            modeler.train_health_model()
            modeler.predict_deviations()
            
            # 獲取統計結果並加入總表
            target_stats = modeler.analyze_disease_groups()
            all_targets_summary.extend(target_stats)
            
            modeler.plot_normative_curves(save_dir=out_dir)
            
    # --- 輸出 Excel 總表 ---
    if all_targets_summary:
        summary_df = pd.DataFrame(all_targets_summary)
        # 重新排列欄位順序，方便閱讀
        cols_order = ['Target', 'Group', 'N', 'Mean_Z_Score', 'Abnormal_Rate', 'Is_High_Risk']
        summary_df = summary_df[cols_order]
        
        excel_path = os.path.join(out_dir, "Normative_Modeling_Summary.xlsx")
        summary_df.to_excel(excel_path, index=False)
        print(f"\n📄 完整統計表格已儲存: {excel_path}")

    print(f"\n✅ 全部完成！請查看資料夾: {out_dir}")