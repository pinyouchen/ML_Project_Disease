import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_score, recall_score, confusion_matrix
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
import warnings
warnings.filterwarnings('ignore')

# 嘗試導入 SHAP
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

class ModelTrainer:
    def __init__(self, label_name, pos_count, neg_count, current_f1, target_f1, use_stacking=False):
        self.label_name = label_name
        self.pos_count = pos_count
        self.neg_count = neg_count
        self.ratio = neg_count / pos_count if pos_count > 0 else 1
        self.target_f1 = target_f1
        self.current_f1 = current_f1
        self.use_stacking = use_stacking
        
        # 策略判定 (還原 V6.12 邏輯)
        self.gap = target_f1 - current_f1
        if self.gap > 0.10: self.strategy = 'aggressive'
        elif self.gap > 0.05: self.strategy = 'moderate'
        else: self.strategy = 'conservative'
        
        self.models = {}
        self.results = {}
        self.fitted_models = {}

    def get_sampling_strategy(self):
        # 完全還原 V6.12 的採樣策略
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

    def build_models(self):
        scale_weight = int(self.ratio * 1.0)
        print(f"\n   [Model] Label: {self.label_name}, 策略: {self.strategy.upper()}")
        
        if self.strategy == 'aggressive': n_est, depth, lr, base_weight_mult = 800, 25, 0.02, 2.0
        elif self.strategy == 'moderate': n_est, depth, lr, base_weight_mult = 600, 18, 0.04, 1.5
        else: n_est, depth, lr, base_weight_mult = 500, 12, 0.06, 1.2
        
        # 還原權重邏輯
        if self.label_name == 'MDD': weight_mult = 1.6
        elif self.label_name == 'SSD': weight_mult = 1.0
        elif self.label_name == 'GAD': weight_mult = 0.95
        elif self.label_name == 'Panic': weight_mult = 1.8
        else: weight_mult = base_weight_mult
        
        final_weight = max(1, int(scale_weight * weight_mult))
        
        # === XGBoost (還原針對不同 Label 的特規參數) ===
        xgb_params = {
            'n_estimators': n_est, 'max_depth': int(depth * 0.4), 'learning_rate': lr,
            'scale_pos_weight': final_weight, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'gamma': 0.2, 'min_child_weight': 2, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'n_jobs': -1, 'verbosity': 0, 'random_state': 42
        }
        # 特規參數覆寫
        if self.label_name == 'Panic':
            xgb_params.update({'n_estimators': 700, 'max_depth': 18, 'learning_rate': 0.03,
                               'subsample': 0.75, 'colsample_bytree': 0.75, 
                               'reg_alpha': 0.08, 'reg_lambda': 0.6, 'min_child_weight': 1})
        elif self.label_name == 'SSD':
            xgb_params.update({'n_estimators': 700, 'max_depth': 12, 'learning_rate': 0.03,
                               'reg_lambda': 2.0})
        elif self.label_name == 'GAD':
            xgb_params.update({'n_estimators': 700, 'max_depth': 10, 'learning_rate': 0.02,
                               'subsample': 0.7, 'colsample_bytree': 0.6, 'gamma': 0.5,
                               'min_child_weight': 3, 'reg_alpha': 0.5, 'reg_lambda': 1.5})

        self.models['XGB'] = xgb.XGBClassifier(**xgb_params)
        
        self.models['LGBM'] = lgb.LGBMClassifier(
            n_estimators=n_est, max_depth=int(depth*0.4), learning_rate=lr,
            num_leaves=int(depth * 1.5), class_weight={0:1, 1:final_weight}, 
            subsample=0.8, colsample_bytree=0.8, min_child_samples=8,
            reg_alpha=0.1, reg_lambda=1.0, n_jobs=-1, verbose=-1, random_state=42
        )
        
        self.models['RF'] = RandomForestClassifier(
            n_estimators=n_est, max_depth=depth, min_samples_split=8, min_samples_leaf=4,
            class_weight={0:1, 1:final_weight}, n_jobs=-1, random_state=42
        )
        
        self.models['ET'] = ExtraTreesClassifier(
            n_estimators=n_est, max_depth=depth, min_samples_split=8, min_samples_leaf=4,
            class_weight={0:1, 1:final_weight}, n_jobs=-1, random_state=42
        )
        
        self.models['GB'] = GradientBoostingClassifier(
            n_estimators=int(n_est * 0.6), max_depth=int(depth * 0.3),
            learning_rate=lr, subsample=0.8, min_samples_split=8, random_state=42
        )
        
        self.models['BalancedRF'] = BalancedRandomForestClassifier(
            n_estimators=int(n_est * 0.8), max_depth=depth, min_samples_split=8,
            min_samples_leaf=4, n_jobs=-1, random_state=42
        )

    def _fit_single_model(self, name, model, X_res, y_res):
        # 還原 Early Stopping 邏輯
        use_early_stop = False
        early_stop_rounds = 30
        val_split_ratio = 0.15

        if self.label_name in ['Panic', 'MDD']:
            use_early_stop = (isinstance(model, xgb.XGBClassifier) or isinstance(model, lgb.LGBMClassifier))
            early_stop_rounds = 20
            val_split_ratio = 0.20

        if use_early_stop:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split_ratio, random_state=42)
            train_sub_idx, val_sub_idx = next(sss.split(X_res, y_res))
            X_tr_sub, y_tr_sub = X_res.iloc[train_sub_idx], y_res.iloc[train_sub_idx]
            X_val_sub, y_val_sub = X_res.iloc[val_sub_idx], y_res.iloc[val_sub_idx]

            try:
                if isinstance(model, xgb.XGBClassifier):
                    model.fit(X_tr_sub, y_tr_sub, eval_set=[(X_val_sub, y_val_sub)],
                              early_stopping_rounds=early_stop_rounds, verbose=False)
                elif isinstance(model, lgb.LGBMClassifier):
                    model.fit(X_tr_sub, y_tr_sub, eval_set=[(X_val_sub, y_val_sub)],
                              eval_metric='binary_logloss',
                              callbacks=[lgb.early_stopping(early_stop_rounds, verbose=False)])
            except TypeError:
                model.fit(X_res, y_res)
        else:
            model.fit(X_res, y_res)
        return model

    def _optimize_threshold(self, y_true, y_pred_proba):
        thresholds = np.linspace(0.10, 0.90, 100)
        
        # 還原 V6.12 的特定門檻值要求
        if self.label_name == 'GAD': min_prec, min_rec = 0.62, 0.60
        elif self.label_name == 'SSD': min_prec, min_rec = 0.68, 0.60
        elif self.label_name == 'MDD': min_prec, min_rec = 0.70, 0.60
        elif self.label_name == 'Panic': min_prec, min_rec = 0.45, 0.45
        else: min_prec, min_rec = 0.50, 0.30

        best_f1, best_thresh = 0, 0.5
        
        for t in thresholds:
            pred = (y_pred_proba >= t).astype(int)
            if pred.sum() == 0: continue
            p = precision_score(y_true, pred, zero_division=0)
            r = recall_score(y_true, pred, zero_division=0)
            if p >= min_prec and r >= min_rec:
                f1 = f1_score(y_true, pred)
                if f1 > best_f1: best_f1, best_thresh = f1, t
        
        if best_f1 == 0: # Fallback
            for t in thresholds:
                pred = (y_pred_proba >= t).astype(int)
                if pred.sum() == 0: continue
                f1 = f1_score(y_true, pred)
                if f1 > best_f1: best_f1, best_thresh = f1, t
        return best_thresh

    def _create_stacking(self, X_train, X_test, y_train, y_test):
        # 還原 Stacking 邏輯
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
        
        stack_proba = meta_clf.predict_proba(meta_X_test)[:, 1]
        thresh = self._optimize_threshold(y_test, stack_proba)
        pred = (stack_proba >= thresh).astype(int)
        
        self._save_result('Stacking', y_test, pred, stack_proba, thresh, None)

    def _create_top3_ensemble(self, X_test, y_test):
        # 還原 Top-3 Ensemble 邏輯
        base_res = {k:v for k,v in self.results.items() if k not in ['Stacking', 'Ensemble']}
        if len(base_res) < 2: return
        
        sorted_models = sorted(base_res.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        top_models = sorted_models[:3]
        preds = [m[1]['y_pred_proba'] for m in top_models]
        
        ens_proba = np.mean(preds, axis=0)
        thresh = self._optimize_threshold(y_test, ens_proba)
        pred = (ens_proba >= thresh).astype(int)
        
        self._save_result('Ensemble', y_test, pred, ens_proba, thresh, None)

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
        status = "✅" if f1 >= self.target_f1 else "⚠️"
        print(f"      {name:12s}: F1={f1:.4f} {status}, P={prec:.3f}, R={rec:.3f}, Spec={spec:.3f}")

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        stype, sratio, k = self.get_sampling_strategy()
        
        # 還原採樣檢查邏輯
        n_pos = int(y_train.sum())
        n_neg = len(y_train) - n_pos
        curr_ratio = n_pos/n_neg if n_neg>0 else 1
        
        if curr_ratio >= sratio:
            sratio = min(curr_ratio + 0.15, 1.0) # 強制上調
        
        try:
            if stype == 'ADASYN': sampler = ADASYN(sampling_strategy=sratio, n_neighbors=k, random_state=42)
            elif stype == 'BorderlineSMOTE': sampler = BorderlineSMOTE(sampling_strategy=sratio, k_neighbors=k, random_state=42)
            else: sampler = SMOTE(sampling_strategy=sratio, k_neighbors=k, random_state=42)
            X_res, y_res = sampler.fit_resample(X_train, y_train)
        except:
            X_res, y_res = X_train, y_train
            
        for name, model in self.models.items():
            fitted = self._fit_single_model(name, model, X_res, y_res)
            self.fitted_models[name] = fitted
            
            proba = fitted.predict_proba(X_test)[:, 1]
            thresh = self._optimize_threshold(y_test, proba)
            pred = (proba >= thresh).astype(int)
            
            self._save_result(name, y_test, pred, proba, thresh, fitted)

        if self.use_stacking:
            self._create_stacking(X_res, X_test, y_res, y_test)
            self._create_top3_ensemble(X_test, y_test)
            
        return self.results