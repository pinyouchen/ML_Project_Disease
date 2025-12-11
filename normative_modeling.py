# normative_modeling_split.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMRegressor

# 設定繪圖風格
sns.set_theme(style="whitegrid", context="talk")
# 設定字體順序：微軟正黑體優先 (解決中文亂碼)，Arial 次之
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

class NormativeModeler:
    def __init__(self, file_path, sheet_name='Data2', 
                 features=['Age', 'Sex', 'BMI'], 
                 target='SDNN',
                 # [回歸] 既然 ID 修好了，我們把心理量表加回來
                 psych_cols=['phq15', 'haq21', 'cabah', 'bdi', 'bai'], 
                 log_transform=True):
        """
        初始化常模建模器 (完整版：含 HRV + 心理量表)
        """
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.features = features
        self.target = target
        self.psych_cols = psych_cols 
        self.log_transform = log_transform
        
        self.df = None
        self.models = {}
        
    def load_data(self):
        """載入資料 (ID 修正 + 心理量表計算)"""
        if not os.path.exists(self.file_path):
            print(f"❌ 找不到檔案: {self.file_path}")
            return False
        
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            
            # ==========================================
            # [絕對修正] 強制使用 Excel 行號 (Index) 作為 ID
            # ==========================================
            self.df['Subject_ID'] = self.df.index
            self.df['Subject_ID'] = self.df['Subject_ID'].astype(str).str.strip().replace(r'\.0$', '', regex=True)

            # 確保必要欄位存在
            cols_needed = self.features + [self.target, 'Health']
            if not all(col in self.df.columns for col in cols_needed):
                print(f"❌ 缺少欄位: {cols_needed}")
                return False
            
            # [回歸] 計算心理量表總分
            available_psych = [c for c in self.psych_cols if c in self.df.columns]
            if available_psych:
                self.df['Psych_Score'] = self.df[available_psych].fillna(0).sum(axis=1)
            else:
                self.df['Psych_Score'] = 0

            # 記錄缺值移除狀況
            n_before = len(self.df)
            self.df = self.df.dropna(subset=cols_needed)
            n_after = len(self.df)
            
            if n_before != n_after:
                print(f"   ℹ️ 提示: 移除 {n_before - n_after} 筆缺值資料")

            # 備份原始值 (Raw)
            self.df[f'{self.target}_Raw'] = self.df[self.target]

            # Log 轉換
            if self.log_transform:
                if (self.df[self.target] < 0).any():
                    pass
                else:
                    self.df[self.target] = np.log1p(self.df[self.target])
            
            return True
        except Exception as e:
            print(f"❌ 載入失敗: {e}")
            return False

    def train_health_model(self):
        """訓練常模"""
        mask_health = self.df['Health'] == 1
        X_health = self.df.loc[mask_health, self.features]
        y_health = self.df.loc[mask_health, self.target]
        
        quantiles = [0.05, 0.50, 0.95]
        names = ['lower', 'median', 'upper']
        
        for q, name in zip(quantiles, names):
            model = LGBMRegressor(objective='quantile', alpha=q, n_estimators=100, learning_rate=0.1, random_state=42, verbose=-1)
            model.fit(X_health, y_health)
            self.models[name] = model

    def predict_deviations(self):
        """計算偏差"""
        X_all = self.df[self.features]
        
        self.df['Pred_Low'] = self.models['lower'].predict(X_all)
        self.df['Pred_Mid'] = self.models['median'].predict(X_all)
        self.df['Pred_High'] = self.models['upper'].predict(X_all)
        
        norm_range = (self.df['Pred_High'] - self.df['Pred_Low'])
        norm_range = norm_range.replace(0, 1e-6)
        estimated_std = norm_range / 3.92 
        
        self.df['Z_Score'] = (self.df[self.target] - self.df['Pred_Mid']) / estimated_std
        self.df['Is_Abnormal_Low'] = self.df['Z_Score'] < -1.96

    # ======================================================
    # Step 2 繪圖 (保留)
    # ======================================================
    def plot_step2_health_normative(self, save_dir=None):
        target_dir = None
        if save_dir:
            target_dir = os.path.join(save_dir, self.target)
            os.makedirs(target_dir, exist_ok=True)

        sex_map = {1: 'Male', 0: 'Female'}

        for sex_val, sex_name in sex_map.items():
            plt.figure(figsize=(12, 7))
            df_sex_health = self.df[(self.df['Sex'] == sex_val) & (self.df['Health'] == 1)].copy()
            if df_sex_health.empty:
                plt.close(); continue

            age_range = np.linspace(df_sex_health['Age'].min(), df_sex_health['Age'].max(), 100)
            X_dummy = pd.DataFrame({'Age': age_range, 'Sex': [sex_val] * len(age_range), 'BMI': [df_sex_health['BMI'].median()] * len(age_range)})

            y_low = self.models['lower'].predict(X_dummy)
            y_mid = self.models['median'].predict(X_dummy)
            y_high = self.models['upper'].predict(X_dummy)

            if self.log_transform and f'{self.target}_Raw' in df_sex_health.columns:
                y_low = np.clip(y_low, -700, 700)
                y_mid = np.clip(y_mid, -700, 700)
                y_high = np.clip(y_high, -700, 700)
                y_low_plot = np.expm1(y_low); y_mid_plot = np.expm1(y_mid); y_high_plot = np.expm1(y_high)
                y_points = df_sex_health[f'{self.target}_Raw']
                y_label = f'{self.target} (original scale)'
            else:
                y_low_plot = y_low; y_mid_plot = y_mid; y_high_plot = y_high
                y_points = df_sex_health[self.target]
                y_label = f'{self.target} Value'

            plt.fill_between(age_range, y_low_plot, y_high_plot, color='green', alpha=0.15, label='Healthy Range')
            plt.plot(age_range, y_mid_plot, color='green', linestyle='--', linewidth=2, label='Median Trend')
            plt.scatter(df_sex_health['Age'], y_points, c='gray', s=25, alpha=0.6, label='Healthy Samples')

            plt.title(f'{self.target} Normative Model (Health only, {sex_name})', fontsize=16)
            plt.xlabel('Age'); plt.ylabel(y_label)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()

            if target_dir:
                plt.savefig(os.path.join(target_dir, f"Step2_Normative_{sex_name}.png"), dpi=300)
                plt.close()
            else:
                plt.show()

    # ======================================================
    # Step 3 繪圖 (保留)
    # ======================================================
    def plot_normative_curves(self, save_dir=None):
        target_dir = None
        if save_dir:
            target_dir = os.path.join(save_dir, self.target)
            os.makedirs(target_dir, exist_ok=True)

        sex_map = {1: 'Male', 0: 'Female'}
        for sex_val, sex_name in sex_map.items():
            plt.figure(figsize=(12, 6))
            age_range = np.linspace(self.df['Age'].min(), self.df['Age'].max(), 100)
            X_dummy = pd.DataFrame({'Age': age_range, 'Sex': [sex_val] * 100, 'BMI': [24] * 100})
            
            y_low = self.models['lower'].predict(X_dummy)
            y_mid = self.models['median'].predict(X_dummy)
            y_high = self.models['upper'].predict(X_dummy)
            
            if self.log_transform:
                y_low = np.clip(y_low, -700, 700); y_mid = np.clip(y_mid, -700, 700); y_high = np.clip(y_high, -700, 700)
                y_low = np.expm1(y_low); y_mid = np.expm1(y_mid); y_high = np.expm1(y_high)

            plt.fill_between(age_range, y_low, y_high, color='green', alpha=0.1, label=f'Healthy Range ({sex_name})')
            plt.plot(age_range, y_mid, color='green', linestyle='--', label='Median Trend')
            
            df_sex = self.df[self.df['Sex'] == sex_val]
            if f'{self.target}_Raw' in df_sex.columns:
                col_plot = f'{self.target}_Raw'
            else:
                col_plot = self.target

            healthy = df_sex[df_sex['Health'] == 1]
            plt.scatter(healthy['Age'], healthy[col_plot], c='gray', s=20, alpha=0.3, label='Healthy')
            abnormal = df_sex[(df_sex['Health'] == 0) & (df_sex['Is_Abnormal_Low'])]
            plt.scatter(abnormal['Age'], abnormal[col_plot], c='red', marker='x', s=60, alpha=0.9, label='Abnormal Pts')
            normal_pt = df_sex[(df_sex['Health'] == 0) & (~df_sex['Is_Abnormal_Low'])]
            plt.scatter(normal_pt['Age'], normal_pt[col_plot], c='blue', s=20, alpha=0.4, label='Normal Pts')
            
            plt.title(f'{self.target} Normative Curve ({sex_name})', fontsize=14)
            plt.xlabel('Age'); plt.ylabel(f'{self.target} Value')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            if target_dir:
                plt.savefig(os.path.join(target_dir, f"Normative_Raw_{sex_name}.png"), dpi=300)
                plt.close()
            else:
                plt.show()

        disease_df = self.df[self.df['Health'] == 0].copy()
        def get_disease_label(row):
            for d in ['Panic', 'MDD', 'GAD', 'SSD']:
                if row.get(d, 0) == 1: return d
            return 'Other'
        disease_df['Disease_Type'] = disease_df.apply(get_disease_label, axis=1)

        plt.figure(figsize=(12, 7))
        plt.axhline(0, color='green', linestyle='--', linewidth=1.5, label='Healthy Median')
        plt.axhline(-1.96, color='red', linestyle='--', linewidth=1.5, label='Lower Limit (95%)')
        plt.axhspan(-1.96, 1.96, color='green', alpha=0.05, label='Normal Range')
        sns.scatterplot(data=disease_df, x='Age', y='Z_Score', hue='Disease_Type', style='Is_Abnormal_Low', palette='deep', s=60, alpha=0.8)
        plt.title(f'{self.target} Deviation Map (Z-Score)', fontsize=16)
        plt.ylabel('Deviation from Norm (Z-Score)'); plt.xlabel('Age'); plt.ylim(-5, 5)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        if target_dir:
            plt.savefig(os.path.join(target_dir, "Normative_ZScore.png"), dpi=300)
            plt.close()
        else:
            plt.show()

    def analyze_disease_groups(self, disease_labels=['SSD', 'MDD', 'Panic', 'GAD']):
        """統計異常率"""
        print(f"\n📊 分析報告: {self.target}")
        print(f"{'Group':<10} | {'Mean Z':<8} | {'Abnormal%':<10}")
        print("-" * 35)
        stats_results = [] 

        health_df = self.df[self.df['Health'] == 1]
        if len(health_df) > 0:
            z_mean = health_df['Z_Score'].mean()
            abn_rate = health_df['Is_Abnormal_Low'].mean()
            print(f"{'Healthy':<10} | {z_mean:>6.2f}   | {abn_rate:>6.1%}")
            stats_results.append({'Target': self.target, 'Group': 'Healthy', 'N': len(health_df), 'Mean_Z_Score': z_mean, 'Abnormal_Rate': abn_rate, 'Is_High_Risk': False})
        
        for label in disease_labels:
            if label in self.df.columns:
                sub_df = self.df[self.df[label] == 1]
                if len(sub_df) > 0:
                    z_mean = sub_df['Z_Score'].mean()
                    abn_rate = sub_df['Is_Abnormal_Low'].mean()
                    is_high_risk = abn_rate > 0.1
                    flag = "🔴" if is_high_risk else "  "
                    print(f"{label:<10} | {z_mean:>6.2f}   | {abn_rate:>6.1%} {flag}")
                    stats_results.append({'Target': self.target, 'Group': label, 'N': len(sub_df), 'Mean_Z_Score': z_mean, 'Abnormal_Rate': abn_rate, 'Is_High_Risk': is_high_risk})
        print("-" * 35)
        return stats_results

    # ======================================================
    # 合併 Step 4 (新增: 臨床亞型定義)
    # ======================================================
    def get_merged_step4_df(self, step1_file_path):
        """讀取 Step 1，合併 Z-Score, Raw 與 Psych_Score，並產生 Clinical_Insight"""
        if not os.path.exists(step1_file_path):
            print(f"⚠️ 找不到 Step 1 檔案: {step1_file_path}")
            return None

        try:
            step1_df = pd.read_excel(step1_file_path)
            
            # 清洗 Step 1 ID
            col_id_s1 = 'Subject_ID' if 'Subject_ID' in step1_df.columns else step1_df.columns[0]
            step1_df[col_id_s1] = step1_df[col_id_s1].astype(str).str.strip().replace(r'\.0$', '', regex=True)
            if col_id_s1 != 'Subject_ID':
                step1_df.rename(columns={col_id_s1: 'Subject_ID'}, inplace=True)

            # 準備合併
            raw_col = f'{self.target}_Raw'
            cols_to_extract = ['Subject_ID', 'Z_Score', 'Psych_Score', raw_col]
            
            df_to_merge = self.df[cols_to_extract].copy()
            rename_dict = {'Z_Score': f'Physio_Z_{self.target}', raw_col: f'{self.target}_Raw'}
            df_to_merge = df_to_merge.rename(columns=rename_dict)
            
            if 'Psych_Score' in step1_df.columns and 'Psych_Score' in df_to_merge.columns:
                df_to_merge = df_to_merge.drop(columns=['Psych_Score'])

            # 合併
            master_df = step1_df.merge(df_to_merge, on='Subject_ID', how='left')

            # 定義分類狀態 TP/TN/FP/FN
            if 'Ground_Truth' in master_df.columns and 'Pred_Label' in master_df.columns:
                conditions = [
                    (master_df['Ground_Truth'] == 1) & (master_df['Pred_Label'] == 1),
                    (master_df['Ground_Truth'] == 0) & (master_df['Pred_Label'] == 0),
                    (master_df['Ground_Truth'] == 0) & (master_df['Pred_Label'] == 1),
                    (master_df['Ground_Truth'] == 1) & (master_df['Pred_Label'] == 0)
                ]
                choices = ['TP', 'TN', 'FP', 'FN']
                master_df['Class_Status'] = np.select(conditions, choices, default='Unknown')

                # ==========================================
                # [新增功能] 根據圖片定義臨床亞型 (Insight)
                # ==========================================
                z_col = f'Physio_Z_{self.target}'
                
                def define_clinical_insight(row):
                    z = row.get(z_col, 0)
                    status = row.get('Class_Status', 'Unknown')
                    
                    if pd.isna(z): return "Unknown"

                    # 1. 偽陰性解析：確診(FN) 但 HRV 正常 (Z > -1) -> 心理主導
                    if status == 'FN' and z > -1:
                        return "Psychological/Cognitive Driven"
                    
                    # 2. 偽陽性解析：未確診(FP) 但 HRV 低 (Z < -1.96) -> 潛在風險
                    if status == 'FP' and z < -1.96:
                        return "Potential Risk Group (Pre-clinical)"
                    
                    # 3. 生理異常亞型：Z < -1.96 (無論分類結果)
                    if z < -1.96:
                        return "Physiological Dysregulation"
                    
                    return "Other / Normal"

                master_df['Clinical_Insight'] = master_df.apply(define_clinical_insight, axis=1)

            return master_df
            
        except Exception as e:
            print(f"❌ 合併失敗: {e}")
            return None

    # ======================================================
    # Step 4 繪圖 (標準圖組)
    # ======================================================
    def plot_step4_visualizations(self, merged_df, save_dir):
        if merged_df is None or merged_df.empty: return
        z_col = f'Physio_Z_{self.target}'
        if z_col not in merged_df.columns: return
        
        plot_df = merged_df.dropna(subset=[z_col, 'Class_Status']).copy()
        if plot_df.empty: return

        target_dir = os.path.join(save_dir, self.target)
        os.makedirs(target_dir, exist_ok=True)
        
        order = ['TP', 'FN', 'TN', 'FP']
        palette = {'TP': '#d62728', 'FN': '#ff7f0e', 'TN': '#2ca02c', 'FP': '#9467bd'}
        
        # 1. Boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Class_Status', y=z_col, data=plot_df, order=order, palette=palette, showfliers=False)
        sns.stripplot(x='Class_Status', y=z_col, data=plot_df, order=order, color='black', alpha=0.3, jitter=True)
        plt.axhline(y=-1.96, color='r', linestyle='--', linewidth=2, label='Abnormal (-1.96)')
        plt.title(f'Triage Boxplot ({self.target})', fontsize=14)
        plt.ylabel('HRV Z-Score')
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, f"Step4_Fig1_Boxplot_{self.target}.png"), dpi=300)
        plt.close()

        # 2. Risk Bar (Updated with specific cuts)
        conditions = [
            (plot_df[z_col] < -1.96),
            (plot_df[z_col] >= -1.96) & (plot_df[z_col] < -1),
            (plot_df[z_col] >= -1)
        ]
        choices = ['Abnormal', 'Borderline', 'Normal']
        plot_df['Risk_Level'] = np.select(conditions, choices, default='Unknown')
        
        risk_counts = plot_df.groupby(['Class_Status', 'Risk_Level']).size().unstack(fill_value=0)
        risk_pct = risk_counts.div(risk_counts.sum(axis=1), axis=0) * 100
        risk_pct = risk_pct.reindex(order)
        risk_cols = ['Abnormal', 'Borderline', 'Normal']
        for c in risk_cols: 
            if c not in risk_pct.columns: risk_pct[c] = 0
        risk_pct = risk_pct[risk_cols]

        risk_pct.plot(kind='bar', stacked=True, color=['#d62728', '#ffcc00', '#2ca02c'], figsize=(10, 6))
        plt.title(f'Risk Stratification ({self.target})', fontsize=14)
        plt.ylabel('Percentage (%)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, f"Step4_Fig2_RiskBar_{self.target}.png"), dpi=300)
        plt.close()

        # 3. Density
        plt.figure(figsize=(10, 6))
        for group in ['TP', 'TN', 'FN', 'FP']:
            subset = plot_df[plot_df['Class_Status'] == group]
            if len(subset) > 3:
                sns.kdeplot(subset[z_col], label=f'{group}', color=palette[group], fill=True, alpha=0.1)
        plt.axvline(x=-1.96, color='r', linestyle='--')
        plt.title(f'Density Plot ({self.target})', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, f"Step4_Fig3_Density_{self.target}.png"), dpi=300)
        plt.close()

    def plot_step4_psych_insight(self, merged_df, save_dir):
        """
        [新增功能] 繪製 Z-Score vs Psych Score
        目的：驗證 FN 是否位於「右上方」(生理正常，心理分高)
        """
        if merged_df is None or merged_df.empty: return
        z_col = f'Physio_Z_{self.target}'
        if z_col not in merged_df.columns or 'Psych_Score' not in merged_df.columns: return

        plot_df = merged_df.dropna(subset=[z_col, 'Psych_Score', 'Class_Status']).copy()
        if plot_df.empty: return

        target_dir = os.path.join(save_dir, self.target)
        palette = {'TP': '#d62728', 'FN': '#ff7f0e', 'TN': '#2ca02c', 'FP': '#9467bd'}

        plt.figure(figsize=(10, 7))
        
        # 繪製散佈圖
        sns.scatterplot(data=plot_df, x=z_col, y='Psych_Score', hue='Class_Status', 
                        palette=palette, style='Class_Status', s=80, alpha=0.7)
        
        # 繪製圖片中的閾值線
        plt.axvline(x=-1.96, color='r', linestyle=':', label='Physio Abnormal (-1.96)')
        plt.axvline(x=-1.0, color='gray', linestyle='--', label='Normal Boundary (-1)')
        
        # 標註區域意義
        plt.text(-3, plot_df['Psych_Score'].max(), 'Physio Driven', color='red', fontsize=10, ha='center')
        plt.text(0, plot_df['Psych_Score'].max(), 'Psych/Cognitive Driven', color='orange', fontsize=10, ha='center')

        plt.title(f'Clinical Insight: Physio vs. Psych ({self.target})', fontsize=14)
        plt.xlabel('Physiological Z-Score (HRV)')
        plt.ylabel('Psychological Score (Questionnaires)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        plt.savefig(os.path.join(target_dir, f"Step4_Fig4_Psych_Insight_{self.target}.png"), dpi=300)
        plt.close()

    # ======================================================
    # 新增方案：拆分繪圖 (Split Focus) - FN / FP 分開畫
    # ======================================================
    def plot_step4_scheme1_split(self, merged_df, save_dir):
        """
        方案一改進版：拆分 FP 與 FN 為兩張獨立圖表，並保留 TN 作為健康基線對照。
        產出：
        1. _Focus_FN.png (強調心理主導)
        2. _Focus_FP.png (強調潛在風險)
        """
        if merged_df is None or merged_df.empty: return
        z_col = f'Physio_Z_{self.target}'
        if z_col not in merged_df.columns or 'Psych_Score' not in merged_df.columns: return

        plot_df = merged_df.dropna(subset=[z_col, 'Psych_Score', 'Class_Status']).copy()
        if plot_df.empty: return

        target_dir = os.path.join(save_dir, self.target)
        os.makedirs(target_dir, exist_ok=True)

        # 取得統一的座標範圍 (讓兩張圖尺度一致，方便比較)
        y_max = plot_df['Psych_Score'].max()
        y_min = plot_df['Psych_Score'].min()
        y_range = y_max - y_min
        xlim = (-5, 5)
        ylim = (y_min - y_range*0.05, y_max + y_range*0.05)

        # 定義繪圖子函數
        def _draw_sub_plot(focus_type, filename_suffix, title_text):
            plt.figure(figsize=(10, 7))
            
            # 1. 背景區域
            plt.axvspan(-10, -1.96, color='red', alpha=0.05)
            plt.axvspan(-1, 10, color='green', alpha=0.05)
            plt.axvline(x=-1.96, color='red', linestyle='--', alpha=0.5)
            plt.axvline(x=-1.0, color='green', linestyle='--', alpha=0.5)

            # 2. 繪製基線 (TN - Healthy Control)
            # 我們需要 TN 存在背景中，讓觀眾知道「正常人」長怎樣
            tn_mask = plot_df['Class_Status'] == 'TN'
            sns.scatterplot(data=plot_df[tn_mask], x=z_col, y='Psych_Score', 
                            color='grey', s=30, alpha=0.15, label='Healthy Baseline (TN)')

            # 3. 繪製焦點群體
            if focus_type == 'FN':
                focus_mask = plot_df['Class_Status'] == 'FN'
                color = '#ff7f0e' # Orange
                marker = 'P' # Plus
                label = 'Missed Cases (FN)'
                # 註解位置：右上方
                plt.text(1.5, y_min + y_range * 0.9, 'Psych-Driven\n(Physio Normal)', 
                         fontsize=12, color='#d35400', ha='center', va='top', fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.5'))
            else: # FP
                focus_mask = plot_df['Class_Status'] == 'FP'
                color = '#9467bd' # Purple
                marker = 'X' # Cross
                label = 'False Alarms (FP)'
                # 註解位置：左下方
                plt.text(-3.5, y_min + y_range * 0.1, 'Physio Risk\n(Pre-clinical)', 
                         fontsize=12, color='#6c3483', ha='center', va='bottom', fontweight='bold',
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.5'))

            sns.scatterplot(data=plot_df[focus_mask], x=z_col, y='Psych_Score', 
                            color=color, marker=marker, s=120, alpha=0.9, linewidth=1.5, edgecolor='w', label=label)

            plt.title(f'{title_text} ({self.target})', fontsize=14)
            plt.xlabel('Physiological Health (HRV Z-Score)')
            plt.ylabel('Psychological Burden (Score)')
            plt.xlim(xlim)
            plt.ylim(ylim)
            plt.legend(loc='upper right')
            plt.tight_layout()
            
            try:
                save_path = os.path.join(target_dir, f"Step4_Scheme1_{filename_suffix}_{self.target}.png")
                plt.savefig(save_path, dpi=300)
                plt.close()
            except Exception as e:
                print(f"無法儲存 {filename_suffix}: {e}")

        # 執行兩次繪圖
        _draw_sub_plot('FN', 'Focus_FN', 'Insight: Psychological Discrepancy')
        _draw_sub_plot('FP', 'Focus_FP', 'Insight: Pre-clinical Risk')

# ==========================================
# 主執行區
# ==========================================
if __name__ == "__main__":
    # 1. 資料路徑 (請修改為您的路徑)
    FILE_PATH = r"D:\FLY114-main\data.xlsx"
    SHEET_NAME = "Data2"
    
    # 2. Step 1 結果路徑 (請修改)
    STEP1_RESULT_PATH = r"D:\ML_Project\runs\Task4_HRV8_D2_20251210_101801\Step1_Predictions_Detail_GAD.xlsx"
    
    hrv_targets = ['SDNN', 'LFHF', 'MEANH', 'TP', 'VLF', 'LF', 'HF', 'NLF']
    out_dir = os.path.join(os.getcwd(), "runs", "normative_analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    final_excel_name = f"Step4_Master_Consolidated_{os.path.basename(STEP1_RESULT_PATH)}"
    final_excel_path = os.path.join(out_dir, final_excel_name)

    all_targets_summary = []

    print(f"🚀 啟動常模分析 (ID修正+心理量表回歸)，結果將寫入: {final_excel_path}")
    
    with pd.ExcelWriter(final_excel_path, engine='openpyxl') as writer:
        for target in hrv_targets:
            # 初始化
            modeler = NormativeModeler(FILE_PATH, SHEET_NAME, target=target, log_transform=True)
            
            if modeler.load_data():
                modeler.train_health_model()
                modeler.predict_deviations()
                
                # 收集統計
                target_stats = modeler.analyze_disease_groups()
                all_targets_summary.extend(target_stats)
                
                # 繪圖 Step 2&3
                modeler.plot_step2_health_normative(save_dir=out_dir)
                modeler.plot_normative_curves(save_dir=out_dir)
                
                # 合併 Step 4
                merged_df = modeler.get_merged_step4_df(STEP1_RESULT_PATH)
                
                if merged_df is not None:
                    sheet_name = target[:30]
                    # 寫入包含 Clinical_Insight 的結果
                    merged_df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"   ✅ 工作表 '{sheet_name}' 已寫入 (含 Clinical_Insight)")
                    
                    # 1. 繪製標準圖表
                    modeler.plot_step4_visualizations(merged_df, save_dir=out_dir)
                    modeler.plot_step4_psych_insight(merged_df, save_dir=out_dir)

                    # 2. [新增] 方案一改進：拆分版本 (FN 一張, FP 一張)
                    modeler.plot_step4_scheme1_split(merged_df, save_dir=out_dir)
                    
                    # (注意：已移除 conceptual 氣泡圖與 error focus 舊版散佈圖)
                else:
                    print(f"   ⚠️ 無法產生 {target} 的合併資料")
            else:
                print(f"   ❌ 資料載入失敗: {target}")

    # 輸出統計摘要
    if all_targets_summary:
        summary_df = pd.DataFrame(all_targets_summary)
        cols_order = ['Target', 'Group', 'N', 'Mean_Z_Score', 'Abnormal_Rate', 'Is_High_Risk']
        cols_exist = [c for c in cols_order if c in summary_df.columns]
        summary_df = summary_df[cols_exist]
        
        summary_path = os.path.join(out_dir, "Normative_Modeling_Summary.xlsx")
        summary_df.to_excel(summary_path, index=False)
        print(f"\n📄 統計總表已儲存: {summary_path}")

    print(f"\n🎉 全部完成！請查看資料夾: {out_dir}")