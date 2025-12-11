import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, RandomizedSearchCV, cross_val_predict, StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix, make_scorer, log_loss
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from scipy.optimize import minimize  # 用於優化 Ensemble 權重
import warnings

warnings.filterwarnings('ignore')

class ModelTrainer:
    def __init__(self, label_name, pos_count, neg_count, current_f1, target_f1, use_stacking=False):
        self.label_name = label_name
        self.pos_count = pos_count
        self.neg_count = neg_count
        self.ratio = neg_count / pos_count if pos_count > 0 else 1
        self.target_f1 = target_f1
        self.use_stacking = use_stacking
        
        self.gap = target_f1 - current_f1
        if self.gap > 0.10: self.strategy = 'aggressive'
        elif self.gap > 0.05: self.strategy = 'moderate'
        else: self.strategy = 'conservative'
        
        self.models = {}
        self.results = {}
        self.fitted_models = {}
        self.best_thresholds = {}

    def get_sampling_strategy(self):
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
        self.models['XGB'] = xgb.XGBClassifier(n_jobs=-1, verbosity=0, random_state=42, use_label_encoder=False)
        self.models['LGBM'] = lgb.LGBMClassifier(n_jobs=-1, verbose=-1, random_state=42)
        self.models['RF'] = RandomForestClassifier(n_jobs=-1, random_state=42)
        self.models['ET'] = ExtraTreesClassifier(n_jobs=-1, random_state=42)
        self.models['GB'] = GradientBoostingClassifier(random_state=42)
        self.models['BalancedRF'] = BalancedRandomForestClassifier(n_jobs=-1, random_state=42)

    def _tune_and_fit(self, name, model, X_res, y_res):
        scale_weight = self.ratio
        param_dist = self.get_param_grid(name, scale_weight)
        
        n_iter = 10 if self.strategy == 'aggressive' else 5
        if self.strategy == 'conservative': n_iter = 5
        
        if not param_dist:
            model.fit(X_res, y_res)
            return model

        print(f"      🔧 Tuning {name} (iter={n_iter})...", end="\r")
        scorer = make_scorer(f1_score)
        
        cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        
        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring=scorer,
            cv=cv_inner,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )
        
        try:
            search.fit(X_res, y_res)
            return search.best_estimator_
        except Exception as e:
            print(f"      ⚠️ Tuning failed for {name}: {e}, using default.")
            model.fit(X_res, y_res)
            return model

    def _optimize_threshold(self, y_true, y_pred_proba):
            # 1. 定義搜尋區間 (保持原樣或依需求調整)
            thresholds = np.linspace(0.20, 0.80, 100)
            
            # 2. 根據疾病設定硬性門檻 (保持原樣)
            if self.label_name == 'GAD': 
                min_prec, min_rec, min_spec = 0.40, 0.50, 0.50
            elif self.label_name == 'Panic':
                min_prec, min_rec, min_spec = 0.20, 0.50, 0.40
            else: 
                min_prec, min_rec, min_spec = 0.30, 0.40, 0.50

            best_score = -1
            best_thresh = 0.5
            
            # 3. 第一階段：有條件優化 (滿足 P/R/S 下的最大 F1)
            for t in thresholds:
                pred = (y_pred_proba >= t).astype(int)
                if pred.sum() == 0: continue 
                
                # 計算混淆矩陣與指標
                # 注意：需確保有引入 confusion_matrix, 或使用 self.metric 計算
                tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                
                # 嚴格篩選
                if prec >= min_prec and rec >= min_rec and spec >= min_spec:
                    score = f1_score(y_true, pred)
                    if score > best_score:
                        best_score = score
                        best_thresh = t
            
            # ============================================================
            # [修改重點] 4. 第二階段備案：無條件 F1 最大化
            # ============================================================
            if best_score == -1:
                print(f"      ⚠️ 嚴格條件未滿足 ({self.label_name})，轉為無條件 F1 最大化...")
                best_f1_fallback = -1
                
                for t in thresholds:
                    pred = (y_pred_proba >= t).astype(int)
                    if pred.sum() == 0: continue
                    
                    # 不管 P/R/S 了，只看 F1
                    score = f1_score(y_true, pred)
                    
                    if score > best_f1_fallback:
                        best_f1_fallback = score
                        best_thresh = t
                
                # 若連這樣都找不到 (例如全 0)，預設回 0.5
                if best_f1_fallback == -1:
                    best_thresh = 0.5

            return best_thresh

    def _find_best_threshold_via_cv(self, model, X_train, y_train):
        try:
            oof_proba = cross_val_predict(model, X_train, y_train, cv=3, method='predict_proba')[:, 1]
            best_thresh = self._optimize_threshold(y_train, oof_proba)
            return best_thresh
        except Exception as e:
            print(f"      ⚠️ CV Thresholding failed: {e}, using 0.5")
            return 0.5

    def _create_stacking(self, X_train, X_test, y_train, y_test):
        if len(self.fitted_models) < 2: return
        
        train_meta = []
        test_meta = []
        
        for name, model in self.fitted_models.items():
            if not hasattr(model, "predict_proba"): continue
            train_meta.append(model.predict_proba(X_train)[:, 1])
            test_meta.append(model.predict_proba(X_test)[:, 1])
            
        if not train_meta: return
        
        meta_X_train = np.vstack(train_meta).T
        meta_X_test  = np.vstack(test_meta).T
        
        cw = {0:1.0, 1:(self.neg_count/self.pos_count)} if self.pos_count>0 else None
        meta_clf = LogisticRegression(max_iter=1000, class_weight=cw, random_state=42)
        
        meta_clf.fit(meta_X_train, y_train)
        meta_thresh = self._find_best_threshold_via_cv(meta_clf, meta_X_train, y_train)
        
        stack_proba = meta_clf.predict_proba(meta_X_test)[:, 1]
        pred = (stack_proba >= meta_thresh).astype(int)
        
        self._save_result('Stacking', y_test, pred, stack_proba, meta_thresh, meta_clf)

    def _create_top3_ensemble(self, X_train, X_test, y_train, y_test):
            base_res = {k:v for k,v in self.results.items() if k not in ['Stacking', 'Ensemble']}
            if len(base_res) < 2: return
            
            sorted_models = sorted(base_res.items(), key=lambda x: x[1]['f1_score'], reverse=True)
            top_models = sorted_models[:3]
            top_names = [m[0] for m in top_models]
            top_objs = [self.fitted_models[name] for name in top_names]
            
            n_models = len(top_names)
            print(f"      🤖 Robust Ensemble using: {top_names} (n={n_models})")

            oof_preds = []
            for model in top_objs:
                try:
                    oof = cross_val_predict(model, X_train, y_train, cv=3, method='predict_proba')[:, 1]
                    oof_preds.append(oof)
                except:
                    oof_preds.append(model.predict_proba(X_train)[:, 1])
            
            oof_preds = np.array(oof_preds).T 
            
            n_trials = 1000
            best_f1 = -1
            best_weights = [1/n_models] * n_models
            best_thresh = 0.5
            
            candidates = np.eye(n_models).tolist()
            candidates.append([1/n_models] * n_models)
            
            np.random.seed(42)
            for _ in range(n_trials):
                w = np.random.dirichlet(np.ones(n_models), size=1)[0]
                candidates.append(w)
                
            for w in candidates:
                if len(w) != n_models: continue
                w_norm = np.array(w) / np.sum(w)
                final_proba = np.average(oof_preds, axis=1, weights=w_norm)
                thresh = self._optimize_threshold(y_train, final_proba)
                pred = (final_proba >= thresh).astype(int)
                score = f1_score(y_train, pred)
                if score > best_f1:
                    best_f1 = score
                    best_weights = w_norm
                    best_thresh = thresh
            
            print(f"      ⚖️  Best Weights (F1 Based): {dict(zip(top_names, np.round(best_weights, 2)))}")

            test_preds = []
            for model in top_objs:
                test_preds.append(model.predict_proba(X_test)[:, 1])
            test_preds = np.array(test_preds).T
            
            ens_proba = np.average(test_preds, axis=1, weights=best_weights)
            pred = (ens_proba >= best_thresh).astype(int)
            
            self._save_result('Ensemble', y_test, pred, ens_proba, best_thresh, None)

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
        stype, sratio, k = self.get_sampling_strategy()
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
            print(f"      [Data] Train: {len(X_train)} -> Resampled: {len(X_res)}")
        except Exception as e:
            print(f"      ⚠️ Resampling failed: {e}, using original data.")
            X_res, y_res = X_train, y_train
            
        for name, model in self.models.items():
            fitted_model = self._tune_and_fit(name, model, X_res, y_res)
            self.fitted_models[name] = fitted_model
            
            best_thresh = self._find_best_threshold_via_cv(fitted_model, X_res, y_res)
            self.best_thresholds[name] = best_thresh
            
            proba = fitted_model.predict_proba(X_test)[:, 1]
            pred = (proba >= best_thresh).astype(int)
            self._save_result(name, y_test, pred, proba, best_thresh, fitted_model)

        if self.use_stacking:
            self._create_stacking(X_res, X_test, y_res, y_test)
            self._create_top3_ensemble(X_res, X_test, y_res, y_test)
            
        return self.results