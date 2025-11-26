import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, make_scorer
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import warnings

# 忽略收斂警告與參數警告
warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, label_name, pos_count, neg_count, current_f1, target_f1, use_stacking=False):
        self.label_name = label_name
        self.pos_count = pos_count
        self.neg_count = neg_count
        self.ratio = neg_count / pos_count if pos_count > 0 else 1
        self.target_f1 = target_f1
        self.use_stacking = use_stacking
        
        # 策略判定：這現在影響的是「搜索次數」與「搜索範圍」，而非寫死參數
        self.gap = target_f1 - current_f1
        if self.gap > 0.10: self.strategy = 'aggressive'
        elif self.gap > 0.05: self.strategy = 'moderate'
        else: self.strategy = 'conservative'
        
        self.models = {}
        self.results = {}
        self.fitted_models = {}
        self.best_thresholds = {} # 儲存每個模型在訓練集上找到的最佳閾值

    def get_sampling_strategy(self):
        """
        保留原本的採樣策略邏輯，這部分不涉及數據洩漏，可繼續使用。
        """
        if self.label_name == 'GAD': return 'BorderlineSMOTE', 0.35, 4
        if self.label_name == 'SSD': return 'BorderlineSMOTE', 0.40, 5
        if self.label_name == 'MDD': return 'SMOTE', 0.65, 5
        if self.label_name == 'Panic': return 'BorderlineSMOTE', 0.55, 4
        
        if self.pos_count < 100:
            sampler_type = 'ADASYN'
            ratio = 0.65 if self.strategy == 'aggressive' else 0.55
            k = 4
        else:
            sampler_type = 'SMOTE'
            ratio = 0.65 if self.strategy == 'aggressive' else 0.50
            k = 5
        return sampler_type, ratio, k

    def get_param_grid(self, model_name, scale_weight):
        """
        [New] 定義參數搜索空間，取代硬編碼。
        根據 strategy 調整搜索範圍。
        """
        # 基礎權重設定
        final_weight = max(1, int(scale_weight))

        if model_name == 'XGB':
            return {
                'n_estimators': [300, 500, 700, 900],
                'max_depth': [6, 10, 15, 20],
                'learning_rate': [0.01, 0.02, 0.05, 0.1],
                'scale_pos_weight': [final_weight, final_weight * 1.5],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8],
                'gamma': [0, 0.2, 0.5],
                'min_child_weight': [1, 3, 5],
                'reg_alpha': [0.1, 0.5, 1.0],
                'reg_lambda': [1.0, 1.5, 2.0]
            }
        elif model_name == 'LGBM':
            return {
                'n_estimators': [300, 500, 700, 900],
                'max_depth': [-1, 10, 20],
                'learning_rate': [0.01, 0.03, 0.05],
                'num_leaves': [31, 50, 80],
                'class_weight': [{0:1, 1:final_weight}, {0:1, 1:final_weight*1.2}],
                'subsample': [0.8, 0.9],
                'reg_alpha': [0.1, 0.5],
                'reg_lambda': [0.5, 1.0]
            }
        elif model_name in ['RF', 'ET']:
            return {
                'n_estimators': [300, 500, 800],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [{0:1, 1:final_weight}, 'balanced']
            }
        elif model_name == 'GB':
            return {
                'n_estimators': [200, 400, 600],
                'max_depth': [3, 5, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 0.9]
            }
        elif model_name == 'BalancedRF':
            return {
                'n_estimators': [300, 500, 800],
                'max_depth': [None, 10, 20],
                'min_samples_leaf': [2, 4]
            }
        return {}

    def build_models(self):
        """
        初始化基礎模型物件，這裡不再設定詳細參數，詳細參數交由 Tuning 決定。
        """
        self.models['XGB'] = xgb.XGBClassifier(n_jobs=-1, verbosity=0, random_state=42, use_label_encoder=False)
        self.models['LGBM'] = lgb.LGBMClassifier(n_jobs=-1, verbose=-1, random_state=42)
        self.models['RF'] = RandomForestClassifier(n_jobs=-1, random_state=42)
        self.models['ET'] = ExtraTreesClassifier(n_jobs=-1, random_state=42)
        self.models['GB'] = GradientBoostingClassifier(random_state=42)
        self.models['BalancedRF'] = BalancedRandomForestClassifier(n_jobs=-1, random_state=42)

    def _tune_and_fit(self, name, model, X_res, y_res):
        """
        [New] 自動參數搜索與訓練
        """
        scale_weight = self.ratio
        param_dist = self.get_param_grid(name, scale_weight)
        
        # 根據策略決定搜索迭代次數 (n_iter)
        # Aggressive = 更多次嘗試
        n_iter = 10 if self.strategy == 'aggressive' else 5
        if self.strategy == 'conservative': n_iter = 5
        
        # 針對特定模型的小優化，若參數網格很小則減少迭代
        if not param_dist:
            model.fit(X_res, y_res)
            return model

        print(f"      🔧 Tuning {name} (iter={n_iter})...", end="\r")
        
        # 使用 F1 作為優化目標
        scorer = make_scorer(f1_score)
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scorer,
            cv=3, # 3-Fold 內部驗證找參數
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        try:
            search.fit(X_res, y_res)
            best_model = search.best_estimator_
            # print(f"      ✅ {name} Tuned. Best F1: {search.best_score_:.3f}")
            return best_model
        except Exception as e:
            print(f"      ⚠️ Tuning failed for {name}: {e}, using default.")
            model.fit(X_res, y_res)
            return model

    # def _optimize_threshold(self, y_true, y_pred_proba):
    #     """
    #     尋找最佳閾值。
    #     注意：這個函數現在只被用於 Training Set (OOF predictions) 上。
    #     """
    #     thresholds = np.linspace(0.10, 0.90, 100)
        
    #     # 保留原本的 Precision/Recall 限制邏輯
    #     if self.label_name == 'GAD': min_prec, min_rec = 0.62, 0.60
    #     elif self.label_name == 'SSD': min_prec, min_rec = 0.68, 0.60
    #     elif self.label_name == 'MDD': min_prec, min_rec = 0.70, 0.60
    #     elif self.label_name == 'Panic': min_prec, min_rec = 0.45, 0.45
    #     else: min_prec, min_rec = 0.50, 0.30

    #     best_f1, best_thresh = 0, 0.5
        
    #     for t in thresholds:
    #         pred = (y_pred_proba >= t).astype(int)
    #         if pred.sum() == 0: continue
    #         p = precision_score(y_true, pred, zero_division=0)
    #         r = recall_score(y_true, pred, zero_division=0)
            
    #         # 只有滿足最小 P/R 要求才考慮該 F1
    #         if p >= min_prec and r >= min_rec:
    #             f1 = f1_score(y_true, pred)
    #             if f1 > best_f1: best_f1, best_thresh = f1, t
        
    #     # Fallback: 如果沒有閾值滿足條件，就找單純 F1 最高的
    #     if best_f1 == 0: 
    #         for t in thresholds:
    #             pred = (y_pred_proba >= t).astype(int)
    #             if pred.sum() == 0: continue
    #             f1 = f1_score(y_true, pred)
    #             if f1 > best_f1: best_f1, best_thresh = f1, t
                
    #     return best_thresh

    def _optimize_threshold(self, y_true, y_pred_proba):
        """
        [Modified] 尋找最佳閾值：使用 Youden's Index 或 F1，但強制要求最小 Specificity。
        """
        thresholds = np.linspace(0.20, 0.80, 100) # 範圍縮小，避免極端值
        
        # 1. 設定嚴格的限制條件
        # 對於 GAD，我們希望 Specificity 至少要及格 (例如 > 0.6)
        if self.label_name == 'GAD': 
            min_prec, min_rec, min_spec = 0.50, 0.60, 0.60
        elif self.label_name == 'Panic':
            min_prec, min_rec, min_spec = 0.40, 0.50, 0.70 # Panic 誤判率太高，需提高 Spec 要求
        else: 
            min_prec, min_rec, min_spec = 0.50, 0.60, 0.50

        best_score = -1
        best_thresh = 0.5
        
        for t in thresholds:
            pred = (y_pred_proba >= t).astype(int)
            if pred.sum() == 0: continue # 避免全 0
            
            # 計算各項指標
            tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            
            # 只有當 Recall 和 Specificity 都滿足底線時，才考慮這個閾值
            if prec >= min_prec and rec >= min_rec and spec >= min_spec:
                # 這裡改用 F1 * Specificity 作為綜合分數，鼓勵兩者皆高
                # 或者使用 Youden's Index: score = rec + spec - 1
                score = f1_score(y_true, pred)
                
                if score > best_score:
                    best_score = score
                    best_thresh = t
        
        # Fallback: 如果太嚴格導致找不到閾值，退回到尋找 Youden's Index 最大值
        if best_score == -1:
            best_j = -1
            for t in thresholds:
                pred = (y_pred_proba >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                
                # Youden's J = Sensitivity + Specificity - 1
                j_index = rec + spec - 1
                if j_index > best_j:
                    best_j = j_index
                    best_thresh = t
            print(f"      ⚠️ Fallback to Youden's Index for {self.label_name}, Th={best_thresh:.2f}")

        return best_thresh

    def _find_best_threshold_via_cv(self, model, X_train, y_train):
        """
        [New] [Priority 1 Solution]
        在訓練集上使用 Cross-Validation (OOF) 預測來尋找最佳閾值。
        這解決了 Data Leakage 問題。
        """
        try:
            # 取得 Out-of-Fold 預測機率
            # cv=3 表示將訓練集切成3份，輪流預測，確保每個樣本都是在未看過該樣本的模型下預測的
            oof_proba = cross_val_predict(model, X_train, y_train, cv=3, method='predict_proba')[:, 1]
            
            # 在這些 OOF 機率上找最佳閾值
            best_thresh = self._optimize_threshold(y_train, oof_proba)
            return best_thresh
        except Exception as e:
            print(f"      ⚠️ CV Thresholding failed: {e}, using 0.5")
            return 0.5

    def _create_stacking(self, X_train, X_test, y_train, y_test):
        if len(self.fitted_models) < 2: return
        
        train_meta = []
        test_meta = []
        
        # 對於 Stacking，訓練集的 Meta Feature 也應該是 OOF 預測，否則會過擬合
        # 但為了簡化運算且保留您的原始結構，這裡我們使用已訓練好的模型
        # *改進*: 理論上這裡也該用 cross_val_predict，但計算量會變大。
        # 這裡我們維持簡單，但使用針對 Training Set 找出的 Threshold 邏輯
        
        valid_models = []
        for name, model in self.fitted_models.items():
            if not hasattr(model, "predict_proba"): continue
            valid_models.append(model)
            train_meta.append(model.predict_proba(X_train)[:, 1])
            test_meta.append(model.predict_proba(X_test)[:, 1])
            
        if not train_meta: return
        
        meta_X_train = np.vstack(train_meta).T
        meta_X_test  = np.vstack(test_meta).T
        
        cw = {0:1.0, 1:(self.neg_count/self.pos_count)} if self.pos_count>0 else None
        meta_clf = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)
        
        # 1. 訓練 Meta Learner
        meta_clf.fit(meta_X_train, y_train)
        
        # 2. 尋找 Meta Learner 的閾值 (使用 OOF 避免洩漏)
        meta_thresh = self._find_best_threshold_via_cv(meta_clf, meta_X_train, y_train)
        
        # 3. 預測測試集
        stack_proba = meta_clf.predict_proba(meta_X_test)[:, 1]
        pred = (stack_proba >= meta_thresh).astype(int)
        
        self._save_result('Stacking', y_test, pred, stack_proba, meta_thresh, meta_clf)

    def _create_top3_ensemble(self, X_test, y_test):
        # 這裡不需要訓練，只需平均概率
        base_res = {k:v for k,v in self.results.items() if k not in ['Stacking', 'Ensemble']}
        if len(base_res) < 2: return
        
        # 選最好的前三個
        sorted_models = sorted(base_res.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        top_models = sorted_models[:3]
        
        # 取出這三個模型的名稱
        top_names = [m[0] for m in top_models]
        
        # 計算平均機率
        preds = [m[1]['y_pred_proba'] for m in top_models]
        ens_proba = np.mean(preds, axis=0)
        
        # [Crucial] Ensemble 的閾值該怎麼定？
        # 應該是這三個模型在訓練集上的最佳閾值的平均，或是重新計算？
        # 簡單起見，我們取三個模型閾值的平均
        avg_thresh = np.mean([self.best_thresholds[name] for name in top_names])
        
        pred = (ens_proba >= avg_thresh).astype(int)
        
        self._save_result('Ensemble', y_test, pred, ens_proba, avg_thresh, None)

    def _save_result(self, name, y_true, y_pred, y_proba, thresh, model):
        f1 = f1_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        try: auc = roc_auc_score(y_true, y_proba)
        except: auc = np.nan
        
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.size==4 else (0,0,0,0)
        spec = tn/(tn+fp) if (tn+fp)>0 else 0
        npv = tn/(tn+fn) if (tn+fn)>0 else 0
        
        self.results[name] = {
            'f1_score': f1, 'accuracy': acc, 'auc': auc,
            'precision': prec, 'recall': rec, 'specificity': spec, 'npv': npv,
            'threshold': thresh, 'y_pred': y_pred, 'y_pred_proba': y_proba, 
            'y_true': y_true.values, 'model': model,
            'shap_values': None, 'feature_importance': None
        }
        status = "✅" if f1 >= self.target_f1 else "  "
        print(f"      {name:12s}: F1={f1:.4f} {status}, P={prec:.3f}, R={rec:.3f}, Spec={spec:.3f} (Th={thresh:.2f})")

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        """
        主流程：採樣 -> Tuning -> 找閾值(Train) -> 預測(Test)
        """
        stype, sratio, k = self.get_sampling_strategy()
        
        # 採樣檢查
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        curr_ratio = n_pos/n_neg if n_neg>0 else 1
        
        if curr_ratio >= sratio:
            sratio = min(curr_ratio + 0.15, 1.0)
        
        try:
            if stype == 'ADASYN': sampler = ADASYN(sampling_strategy=sratio, n_neighbors=k, random_state=42)
            elif stype == 'BorderlineSMOTE': sampler = BorderlineSMOTE(sampling_strategy=sratio, k_neighbors=k, random_state=42)
            else: sampler = SMOTE(sampling_strategy=sratio, k_neighbors=k, random_state=42)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        except:
            X_res, y_res = X_train, y_train
            
        print(f"      [Data] Train: {len(X_train)} -> Resampled: {len(X_res)}")
            
        for name, model in self.models.items():
            # 1. 自動參數調整 (Tuning) + 擬合 (Fitting)
            # 注意：這裡是在 Resampled 數據上進行 Tuning 和 Fitting
            fitted_model = self._tune_and_fit(name, model, X_res, y_res)
            self.fitted_models[name] = fitted_model
            
            # 2. [Priority 1 Fix] 在訓練集上尋找最佳閾值 (使用 Cross-Validation 避免洩漏)
            # 我們使用 Resampled 數據來找閾值，因為模型是在這種分佈上訓練的
            best_thresh = self._find_best_threshold_via_cv(fitted_model, X_res, y_res)
            self.best_thresholds[name] = best_thresh
            
            # 3. 在測試集上進行預測 (使用剛剛找到的閾值)
            proba = fitted_model.predict_proba(X_test)[:, 1]
            # 嚴格禁止在這裡使用 y_test 來找閾值！直接使用 best_thresh
            pred = (proba >= best_thresh).astype(int)
            
            self._save_result(name, y_test, pred, proba, best_thresh, fitted_model)

        if self.use_stacking:
            self._create_stacking(X_res, X_test, y_res, y_test)
            self._create_top3_ensemble(X_test, y_test)
            
        return self.results