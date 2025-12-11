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
        
        # 預設特徵群組
        self.basic_features = ['Age', 'Sex', 'BMI'] 
        self.hrv_4_features = ['SDNN', 'LF', 'HF', 'LFHF'] # Baseline
        self.hrv_8_features = ['SDNN', 'LF', 'HF', 'LFHF', 'MEANH', 'VLF', 'NLF', 'TP'] # Advanced Data2
        self.psych_features = ['phq15', 'haq21', 'cabah', 'bdi', 'bai'] # Advanced Data2
        self.clinical_features = ['DM', 'TCA', 'MARTA'] # Data1 獨有，需補0
        self.label_names = ['Health', 'SSD', 'MDD', 'Panic', 'GAD']
        self.log_hrv_cols = ['LF', 'HF', 'LFHF', 'TP', 'VLF', 'NLF']
        self.log_engineered_cols = ['HRV_Mean', 'LF_HF_Ratio','HF_TP_Ratio']

    def load_data(self):
            try:
                if self.file_path.lower().endswith('.csv'):
                    print(f"📂 偵測到 CSV 格式，正在讀取: {self.file_path}")
                    self.df = pd.read_csv(self.file_path)
                else:
                    print(f"📂 正在讀取 Excel: {self.file_path} (Sheet: {self.sheet_name})")
                    self.df = pd.read_excel(self.file_path, sheet_name=self.sheet_name)

                # 簡單前處理：性別轉數值
                if 'Sex' in self.df.columns and self.df['Sex'].dtype == 'O':
                    self.df['Sex'] = self.df['Sex'].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0, '1': 1, '0': 0})

                print(f"✓ 載入成功: {self.df.shape[0]} 筆")
                return True
            except Exception as e:
                print(f"❌ 載入失敗: {e}")
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
            # 針對 HRV 特徵若為 0 視為缺失
            hrv_candidates = self.hrv_4_features + self.hrv_8_features
            for col in [c for c in hrv_candidates if c in Xp.columns]:
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
        
        # Clinical Features 補 0
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

        # KNN Imputation
        knn_f = self._numeric_feature_list_for_outlier(X_train_p)
        if len(knn_f) > 0:
            if fit or self.knn_imputer is None:
                self.knn_imputer = KNNImputer(n_neighbors=5)
                if not X_train_p[knn_f].isnull().all().all():
                    X_train_p[knn_f] = self.knn_imputer.fit_transform(X_train_p[knn_f])
            else:
                X_train_p[knn_f] = self.knn_imputer.transform(X_train_p[knn_f])
            
            if X_test_p is not None:
                X_test_p[knn_f] = self.knn_imputer.transform(X_test_p[knn_f])
        
        # Median Fill as fallback
        X_train_p.fillna(X_train_p.median(numeric_only=True), inplace=True)
        if X_test_p is not None:
            X_test_p.fillna(X_train_p.median(numeric_only=True), inplace=True)

        # StandardScaler
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


# ==========================================
# Task 1: Baseline (4 HRV + Demo)
# ==========================================
class ProcessorBaseline4(BaseProcessor):
    def prepare_features_and_labels(self):
        features = self.basic_features + self.hrv_4_features
        available = [f for f in features if f in self.df.columns]
        self.X = self.df[available].copy()
        
        if 'LF' in self.X and 'HF' in self.X:
            self.X['LF_HF_Ratio'] = self.X['LF'] / (self.X['HF'] + 1e-6)
        # [新增] 2. 計算 HRV_Mean (使用 hrv_4_features)
        hrv_cols_present = [c for c in self.hrv_4_features if c in self.X.columns]
        if len(hrv_cols_present) >= 3:
            self.X['HRV_Mean'] = self.X[hrv_cols_present].mean(axis=1)
            
        for label in self.label_names:
            if label in self.df.columns: self.y_dict[label] = self.df[label].copy()
        return True

# ==========================================
# Task 3: Comparison (Psych Only + Demo)
# ==========================================
class ProcessorPsych(BaseProcessor):
    def prepare_features_and_labels(self):
        features = self.basic_features + self.psych_features
        available = [f for f in features if f in self.df.columns]
        self.X = self.df[available].copy()
        
        for label in self.label_names:
            if label in self.df.columns: self.y_dict[label] = self.df[label].copy()
        return True

# ==========================================
# Task 4: Advanced HRV (8 HRV + Demo)
# ==========================================
class ProcessorHRV8(BaseProcessor):
    def prepare_features_and_labels(self):
        features = self.basic_features + self.hrv_8_features
        available = [f for f in features if f in self.df.columns]
        self.X = self.df[available].copy()
        
        if 'LF' in self.X and 'HF' in self.X:
            self.X['LF_HF_Ratio'] = self.X['LF'] / (self.X['HF'] + 1e-6)

        # [新增] 2. 計算 HF/TP Ratio
        if 'HF' in self.X.columns and 'TP' in self.X.columns:
            self.X['HF_TP_Ratio'] = self.X['HF'] / (self.X['TP'] + 1e-6)

        # [新增] 3. 計算 HRV_Mean (使用 hrv_8_features)
        hrv_cols_present = [c for c in self.hrv_8_features if c in self.X.columns]
        if len(hrv_cols_present) >= 3:
            self.X['HRV_Mean'] = self.X[hrv_cols_present].mean(axis=1)
        
        for label in self.label_names:
            if label in self.df.columns: self.y_dict[label] = self.df[label].copy()
        return True

# ==========================================
# Task 5: Data2 Full (8 HRV + Psych + Demo)
# ==========================================
class ProcessorData2Full(BaseProcessor):
    def prepare_features_and_labels(self):
        features = self.basic_features + self.hrv_8_features + self.psych_features
        available = [f for f in features if f in self.df.columns]
        self.X = self.df[available].copy()
        
        if 'LF' in self.X and 'HF' in self.X:
            self.X['LF_HF_Ratio'] = self.X['LF'] / (self.X['HF'] + 1e-6)

        # [新增] 2. 計算 HF/TP Ratio
        if 'HF' in self.X.columns and 'TP' in self.X.columns:
            self.X['HF_TP_Ratio'] = self.X['HF'] / (self.X['TP'] + 1e-6)

        # [新增] 3. 計算 HRV_Mean (使用 hrv_8_features)
        hrv_cols_present = [c for c in self.hrv_8_features if c in self.X.columns]
        if len(hrv_cols_present) >= 3:
            self.X['HRV_Mean'] = self.X[hrv_cols_present].mean(axis=1)
            
        for label in self.label_names:
            if label in self.df.columns: self.y_dict[label] = self.df[label].copy()
        return True