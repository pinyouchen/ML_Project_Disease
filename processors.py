import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

class BaseProcessor:
    def __init__(self, file_path, sheet_name, iqr_multiplier=3.0, treat_zero_as_missing_in_hrv=True):
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.iqr_multiplier = iqr_multiplier
        self.treat_zero_as_missing_in_hrv = treat_zero_as_missing_in_hrv
        self.df = None
        self.X = None
        self.y_dict = {}
        self.knn_imputer = None
        self.scaler = None
        self.outlier_bounds_ = None
        
        # 預設欄位定義 (子類別可覆寫)
        self.hrv_features = []
        self.basic_features = ['Age', 'Sex', 'BMI']
        self.psych_features = []
        self.label_names = ['Health', 'SSD', 'MDD', 'Panic', 'GAD']
        self.log_hrv_cols = []
        self.log_engineered_cols = []
        self.clinical_features = []

    def load_data(self):
        try:
            self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
            print(f"✓ 載入: {self.df.shape[0]} 筆（工作表：{self.sheet_name}）")
            return True
        except Exception as e:
            print(f"❌ {e}")
            return False

    def _compute_iqr_bounds(self, s, k):
        q1 = s.quantile(0.25); q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr) or iqr == 0:
            lower = s.quantile(0.001); upper = s.quantile(0.999)
        else:
            lower = q1 - k * iqr; upper = q3 + k * iqr
        return float(lower), float(upper)

    def _numeric_feature_list_for_outlier(self, X_frame):
        # 簡單邏輯：所有數值型欄位除了 Sex
        return [c for c in X_frame.columns if c != 'Sex' and pd.api.types.is_numeric_dtype(X_frame[c])]

    def _fit_outlier_bounds(self, X_train):
        num_cols = self._numeric_feature_list_for_outlier(X_train)
        bounds = {}
        for col in num_cols:
            s = pd.to_numeric(X_train[col], errors='coerce')
            lower, upper = self._compute_iqr_bounds(s.dropna(), self.iqr_multiplier)
            bounds[col] = (lower, upper)
        self.outlier_bounds_ = bounds

    def _apply_outlier_to_nan(self, X_frame):
        if not self.outlier_bounds_: return X_frame
        Xp = X_frame.copy()
        if self.treat_zero_as_missing_in_hrv:
            for col in [c for c in self.hrv_features if c in Xp.columns]:
                s = pd.to_numeric(Xp[col], errors='coerce')
                Xp.loc[s == 0, col] = np.nan
        for col, (lb, ub) in self.outlier_bounds_.items():
            if col in Xp.columns:
                s = pd.to_numeric(Xp[col], errors='coerce')
                Xp.loc[(s < lb) | (s > ub), col] = np.nan
        return Xp

    def _apply_log1p(self, X_frame):
        Xp = X_frame.copy()
        for col in self.log_hrv_cols + self.log_engineered_cols:
            if col in Xp.columns:
                s = pd.to_numeric(Xp[col], errors='coerce')
                neg_mask = s < 0
                if neg_mask.any(): Xp.loc[neg_mask, col] = np.nan
                Xp[col] = np.log1p(Xp[col])
        return Xp

    def impute_and_scale(self, X_train, X_test=None, fit=True):
        X_train_p = X_train.copy()
        X_test_p = X_test.copy() if X_test is not None else None
        
        # 先針對 Clinical Features 補 0
        for f in self.clinical_features:
            if f in X_train_p.columns:
                X_train_p[f].fillna(0, inplace=True)
                if X_test_p is not None and f in X_test_p.columns:
                    X_test_p[f].fillna(0, inplace=True)

        if fit: self._fit_outlier_bounds(X_train_p)
        X_train_p = self._apply_outlier_to_nan(X_train_p)
        if X_test_p is not None: X_test_p = self._apply_outlier_to_nan(X_test_p)
        
        X_train_p = self._apply_log1p(X_train_p)
        if X_test_p is not None: X_test_p = self._apply_log1p(X_test_p)

        knn_f = self._numeric_feature_list_for_outlier(X_train_p)
        if len(knn_f) > 0:
            if fit or self.knn_imputer is None:
                self.knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
                if not X_train_p[knn_f].isnull().all().all():
                    X_train_p[knn_f] = self.knn_imputer.fit_transform(X_train_p[knn_f])
            else:
                X_train_p[knn_f] = self.knn_imputer.transform(X_train_p[knn_f])
            
            if X_test_p is not None:
                X_test_p[knn_f] = self.knn_imputer.transform(X_test_p[knn_f])
        
        X_train_p.fillna(X_train_p.median(numeric_only=True), inplace=True)
        if X_test_p is not None:
            X_test_p.fillna(X_train_p.median(numeric_only=True), inplace=True)

        cols = X_train_p.columns.tolist()
        num_cols = [c for c in cols if c != 'Sex' and pd.api.types.is_numeric_dtype(X_train_p[c])]
        other_cols = [c for c in cols if c not in num_cols]
        
        if fit or self.scaler is None:
            self.scaler = StandardScaler()
            if len(num_cols) > 0:
                X_train_num = pd.DataFrame(self.scaler.fit_transform(X_train_p[num_cols]), columns=num_cols, index=X_train_p.index)
            else:
                X_train_num = pd.DataFrame(index=X_train_p.index)
        else:
            if len(num_cols) > 0:
                X_train_num = pd.DataFrame(self.scaler.transform(X_train_p[num_cols]), columns=num_cols, index=X_train_p.index)
            else:
                X_train_num = pd.DataFrame(index=X_train_p.index)
            
        X_train_s = pd.concat([X_train_num, X_train_p[other_cols]], axis=1)[cols]
        
        if X_test_p is not None:
            if len(num_cols) > 0:
                X_test_num = pd.DataFrame(self.scaler.transform(X_test_p[num_cols]), columns=num_cols, index=X_test_p.index)
            else:
                X_test_num = pd.DataFrame(index=X_test_p.index)
            X_test_s = pd.concat([X_test_num, X_test_p[other_cols]], axis=1)[cols]
            return X_train_s, X_test_s
            
        return X_train_s


# 1. HRV 任務 (完整 8 個 HRV)
class ProcessorHRV(BaseProcessor):
    def __init__(self, file_path, sheet_name='Data2'):
        super().__init__(file_path, sheet_name)
        self.hrv_features = ['SDNN', 'LF', 'HF', 'LFHF', 'MEANH', 'TP', 'VLF', 'NLF']
        self.log_hrv_cols = ['LF', 'HF', 'LFHF', 'TP', 'VLF', 'NLF']
        self.log_engineered_cols = ['HRV_Mean', 'LF_HF_Ratio']

    def prepare_features_and_labels(self):
        all_features = self.basic_features + self.hrv_features
        available = [f for f in all_features if f in self.df.columns]
        self.X = self.df[available].copy()
        
        hrv_cols = [c for c in self.hrv_features if c in self.X.columns]
        if len(hrv_cols) >= 3: self.X['HRV_Mean'] = self.X[hrv_cols].mean(axis=1)
        if 'LF' in self.X.columns and 'HF' in self.X.columns:
            self.X['LF_HF_Ratio'] = self.X['LF'] / (self.X['HF'] + 1e-6)
            self.X['LF_HF_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

        for label in self.label_names:
            if label in self.df.columns: self.y_dict[label] = self.df[label].copy()
        return len(self.y_dict) > 0


# 2. Psych 任務
class ProcessorPsych(BaseProcessor):
    def __init__(self, file_path, sheet_name='Data2'):
        super().__init__(file_path, sheet_name)
        self.psych_features = ['phq15', 'haq21', 'cabah', 'bdi', 'bai']
    
    def prepare_features_and_labels(self):
        all_features = self.basic_features + self.psych_features
        available = [f for f in all_features if f in self.df.columns]
        self.X = self.df[available].copy()
        for label in self.label_names:
            if label in self.df.columns: self.y_dict[label] = self.df[label].copy()
        return len(self.y_dict) > 0


# 3. Baseline All (HRV 8 + Psych + Demo)
class ProcessorBaselineAll(BaseProcessor):
    def __init__(self, file_path, sheet_name='Data2'):
        super().__init__(file_path, sheet_name)
        self.hrv_features = ['SDNN', 'LF', 'HF', 'LFHF', 'MEANH', 'TP', 'VLF', 'NLF']
        self.psych_features = ['phq15', 'haq21', 'cabah', 'bdi', 'bai']
        self.log_hrv_cols = ['LF', 'HF', 'LFHF', 'TP', 'VLF', 'NLF']
        self.log_engineered_cols = ['HRV_Mean', 'LF_HF_Ratio']
        
    def prepare_features_and_labels(self):
        all_features = self.basic_features + self.hrv_features + self.psych_features
        available = [f for f in all_features if f in self.df.columns]
        self.X = self.df[available].copy()
        
        hrv_cols = [c for c in self.hrv_features if c in self.X.columns]
        if len(hrv_cols) >= 3: self.X['HRV_Mean'] = self.X[hrv_cols].mean(axis=1)
        if 'LF' in self.X.columns and 'HF' in self.X.columns:
            self.X['LF_HF_Ratio'] = self.X['LF'] / (self.X['HF'] + 1e-6)
            self.X['LF_HF_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            
        for label in self.label_names:
            if label in self.df.columns: self.y_dict[label] = self.df[label].copy()
        return len(self.y_dict) > 0


# 4. Full V6.2 (完整特徵)
class ProcessorFullV62(BaseProcessor):
    def __init__(self, file_path, sheet_name='Merged_Sheet'):
        super().__init__(file_path, sheet_name)
        self.hrv_features = ['MEANH', 'LF', 'HF', 'NLF', 'SC', 'FT', 'RSA', 'TP', 'VLF']
        self.clinical_features = ['DM', 'TCA', 'MARTA']
        self.psych_features = ['phq15', 'haq21', 'cabah', 'bdi', 'bai']
        self.log_hrv_cols = ['LF', 'HF', 'TP', 'VLF', 'SC', 'RSA']
        self.log_engineered_cols = ['HRV_Mean', 'LF_HF_Ratio', 'bai_log', 'phq15_log', 'bdi_log', 'cabah_log']

    def prepare_features_and_labels(self):
        all_features = (self.basic_features + self.hrv_features + 
                        self.clinical_features + self.psych_features)
        available = [f for f in all_features if f in self.df.columns]
        self.X = self.df[available].copy()
        
        print(f"\n🔨 特徵工程 (V6.2 完整版)...")
        
        # 1. 基礎: Age * BMI
        if 'Age' in self.X.columns and 'BMI' in self.X.columns:
            self.X['Age_BMI'] = self.X['Age'] * self.X['BMI']
        
        # 2. HRV 相關
        hrv_cols = [c for c in self.hrv_features if c in self.X.columns]
        if len(hrv_cols) >= 3: self.X['HRV_Mean'] = self.X[hrv_cols].mean(axis=1)
        if 'LF' in self.X.columns and 'HF' in self.X.columns:
            self.X['LF_HF_Ratio'] = self.X['LF'] / (self.X['HF'] + 1e-6)
            self.X['LF_HF_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
            self.X['Sympathetic_Index'] = self.X['LF'] / (self.X['LF'] + self.X['HF'] + 1e-6)

        # 3. Psych Sum
        psych_cols = [c for c in self.psych_features if c in self.X.columns]
        if len(psych_cols) >= 3:
            self.X['Psych_Sum'] = self.X[psych_cols].sum(axis=1)

        # 4. Panic 特徵
        if 'bai' in self.X.columns:
            self.X['bai_log'] = np.log1p(self.X['bai'])
        if 'bai' in self.X.columns and 'HRV_Mean' in self.X.columns:
            self.X['Panic_Risk'] = self.X['bai'] / (self.X['HRV_Mean'] + 1e-6)

        # 5. SSD, MDD, GAD 特徵
        if 'phq15' in self.X.columns:
            self.X['phq15_log'] = np.log1p(self.X['phq15'])
        if 'phq15' in self.X.columns and 'Age' in self.X.columns:
            self.X['Somatic_Age_Interaction'] = self.X['phq15'] * self.X['Age']

        if 'bdi' in self.X.columns:
            self.X['bdi_log'] = np.log1p(self.X['bdi'])
        if 'bdi' in self.X.columns and 'HRV_Mean' in self.X.columns:
            self.X['Depression_HRV_Ratio'] = self.X['bdi'] / (self.X['HRV_Mean'] + 1e-6)
        
        if 'cabah' in self.X.columns:
            self.X['cabah_log'] = np.log1p(self.X['cabah'])
        if 'cabah' in self.X.columns and 'TP' in self.X.columns:
            self.X['Anxiety_TP_Ratio'] = self.X['cabah'] / (self.X['TP'] + 1e-6)

        for label in self.label_names:
            if label in self.df.columns: self.y_dict[label] = self.df[label].copy()
        
        print(f"✓ 總特徵數量: {self.X.shape[1]}")
        return len(self.y_dict) > 0


# 5. Baseline (原始 4 個 HRV) - 繼承 BaseProcessor
class DataProcessorBaseline(BaseProcessor):
    """
    Baseline: 對應 test2_data2_binary.py 的設定
    只使用 4 個基礎 HRV 特徵 (SDNN, LF, HF, LFHF) + Demo
    """
    def __init__(self, file_path, sheet_name='Data2'):
        super().__init__(file_path, sheet_name)
        # 🔥 修正：只定義 4 個 HRV 特徵
        self.hrv_features = ['SDNN', 'LF', 'HF', 'LFHF']
        self.log_hrv_cols = ['LF', 'HF', 'LFHF']
        self.log_engineered_cols = ['HRV_Mean', 'LF_HF_Ratio']

    def prepare_features_and_labels(self):
        all_features = self.basic_features + self.hrv_features
        available = [f for f in all_features if f in self.df.columns]
        self.X = self.df[available].copy()
        
        # Feature Engineering
        hrv_cols = [c for c in self.hrv_features if c in self.X.columns]
        if len(hrv_cols) >= 3: self.X['HRV_Mean'] = self.X[hrv_cols].mean(axis=1)
        if 'LF' in self.X.columns and 'HF' in self.X.columns:
            self.X['LF_HF_Ratio'] = self.X['LF'] / (self.X['HF'] + 1e-6)
            self.X['LF_HF_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)

        for label in self.label_names:
            if label in self.df.columns: self.y_dict[label] = self.df[label].copy()
        return len(self.y_dict) > 0