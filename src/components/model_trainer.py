import os
import sys
import numpy as np
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score, classification_report, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    threshold_file_path     = os.path.join("artifacts", "threshold.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def find_best_threshold(self, y_true, y_proba):
        best_thresh, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.60, 0.01):
            f1 = f1_score(y_true, (y_proba >= t).astype(int), zero_division=0)
            if f1 > best_f1:
                best_f1, best_thresh = f1, t
        return round(float(best_thresh), 2)

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting train and test arrays")
            X_train, y_train = train_array[:, :-1], train_array[:, -1].astype(int)
            X_test,  y_test  = test_array[:,  :-1], test_array[:,  -1].astype(int)

            logging.info(f"Train size: {X_train.shape} | Test size: {X_test.shape}")
            logging.info(f"Train delay rate: {y_train.mean():.1%} | Test delay rate: {y_test.mean():.1%}")

            classes = np.array([0, 1])
            weights = compute_class_weight('balanced', classes=classes, y=y_train)
            cw_dict = {0: weights[0], 1: weights[1]}
            n_neg   = (y_train == 0).sum()
            n_pos   = (y_train == 1).sum()
            spw     = n_neg / n_pos

            logging.info(f"Class weights: {cw_dict} | scale_pos_weight: {spw:.1f}")

            models = {
                "Random Forest": RandomForestClassifier(
                    class_weight=cw_dict, random_state=42, n_jobs=1
                ),
                "XGBoost": XGBClassifier(
                    scale_pos_weight=spw, eval_metric='logloss',
                    random_state=42, n_jobs=1, device='cpu', tree_method='hist'
                ),
                "LightGBM": LGBMClassifier(
                    scale_pos_weight=spw,
                    random_state=42, n_jobs=1, verbose=-1
                ),
                "CatBoost": CatBoostClassifier(
                    auto_class_weights='Balanced',
                    random_seed=42, verbose=0
                )
            }

            params = {
                "Random Forest": {
                    'n_estimators': [100, 300],
                    'max_depth'   : [8, 12]
                },
                "XGBoost": {
                    'n_estimators' : [100, 300],
                    'max_depth'    : [4, 6],
                    'learning_rate': [0.05, 0.1]
                },
                "LightGBM": {
                    'n_estimators' : [100, 300],
                    'num_leaves'   : [31, 63],
                    'learning_rate': [0.05, 0.1]
                },
                "CatBoost": {
                    'iterations'   : [100, 300],
                    'depth'        : [4, 6],
                    'learning_rate': [0.05, 0.1]
                }
            }

            logging.info("Starting model evaluation with hyperparameter tuning")
            model_report = evaluate_models(
                X_train=X_train, y_train=y_train,
                X_test=X_test,   y_test=y_test,
                models=models,   param=params
            )

            logging.info(f"Model report: {model_report}")

            best_model_score = max(model_report.values())
            best_model_name  = max(model_report, key=model_report.get)
            best_model       = models[best_model_name]

            logging.info(f"Best model: {best_model_name} | ROC-AUC: {best_model_score:.4f}")

            if best_model_score < 0.60:
                raise CustomException("No acceptable model found — ROC-AUC below 0.60")

            # Fix 1 — dynamic threshold instead of hardcoded 0.59
            y_train_proba = best_model.predict_proba(X_train)[:, 1]
            best_thresh   = self.find_best_threshold(y_train, y_train_proba)
            logging.info(f"Best threshold (tuned): {best_thresh}")

            # Fix 2 — save threshold so predict_pipeline can load it
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            save_object(
                file_path=self.model_trainer_config.threshold_file_path,
                obj=best_thresh
            )
            logging.info(f"Saved model.pkl and threshold.pkl")

            y_test_proba = best_model.predict_proba(X_test)[:, 1]
            y_test_pred  = (y_test_proba >= best_thresh).astype(int)
            final_auc    = roc_auc_score(y_test, y_test_proba)

            logging.info(f"Final Test ROC-AUC: {final_auc:.4f}")
            logging.info(f"\n{classification_report(y_test, y_test_pred, target_names=['On time', 'Delayed'], zero_division=0)}")

            return final_auc

        except Exception as e:
            raise CustomException(e, sys)