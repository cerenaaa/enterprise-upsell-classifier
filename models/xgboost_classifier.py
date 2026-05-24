"""
XGBoost binary classifier for enterprise upsell prediction.
Includes Optuna tuning, probability calibration, and SHAP explanations.
"""
import json
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import shap
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

from features.usage_signals import add_usage_features, get_feature_cols

optuna.logging.set_verbosity(optuna.logging.WARNING)

CATEGORICAL = ["industry", "region", "segment"]
NUMERIC = None  # resolved at runtime from get_feature_cols()


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ])


class UpsellClassifier:
    def __init__(self, n_trials: int = 50, cv_folds: int = 5, random_state: int = 42):
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.pipeline = None
        self.best_params = None
        self.feature_names_ = None

    def _prepare(self, df: pd.DataFrame):
        df = add_usage_features(df)
        numeric_cols = [c for c in get_feature_cols() if c in df.columns]
        X = df[numeric_cols + CATEGORICAL]
        return X, numeric_cols

    def _objective(self, trial, X, y):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 4.0),
            "random_state": self.random_state,
        }
        _, numeric_cols = self._prepare(pd.DataFrame(columns=X.columns))
        prep = build_preprocessor(numeric_cols if numeric_cols else list(X.select_dtypes("number").columns))
        clf = xgb.XGBClassifier(**params, eval_metric="auc", verbosity=0)
        pipe = Pipeline([("prep", prep), ("clf", clf)])
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
        return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1).mean()

    def tune(self, df: pd.DataFrame, y: pd.Series):
        X, _ = self._prepare(df)
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: self._objective(t, X, y), n_trials=self.n_trials)
        self.best_params = study.best_params
        print(f"Best CV AUC: {study.best_value:.4f}")
        return self

    def fit(self, df: pd.DataFrame, y: pd.Series):
        X, numeric_cols = self._prepare(df)
        params = self.best_params or {
            "n_estimators": 300, "max_depth": 5, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8, "random_state": self.random_state,
        }
        prep = build_preprocessor(numeric_cols)
        base = xgb.XGBClassifier(**params, eval_metric="auc", verbosity=0)
        calibrated = CalibratedClassifierCV(base, method="isotonic", cv=3)
        self.pipeline = Pipeline([("prep", prep), ("clf", calibrated)])
        self.pipeline.fit(X, y)
        self.feature_names_ = numeric_cols + CATEGORICAL
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X, _ = self._prepare(df)
        return self.pipeline.predict_proba(X)[:, 1]

    def evaluate(self, df: pd.DataFrame, y: pd.Series) -> dict:
        proba = self.predict_proba(df)
        metrics = {
            "roc_auc": roc_auc_score(y, proba),
            "avg_precision": average_precision_score(y, proba),
            "brier_score": brier_score_loss(y, proba),
        }
        print(f"ROC-AUC={metrics['roc_auc']:.4f} | AvgPrec={metrics['avg_precision']:.4f} | Brier={metrics['brier_score']:.4f}")
        return metrics

    def decile_lift(self, df: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Compute lift table by score decile — key metric for sales prioritization."""
        proba = self.predict_proba(df)
        results = pd.DataFrame({"score": proba, "label": y})
        results["decile"] = pd.qcut(results["score"], q=10, labels=False, duplicates="drop")
        decile_table = (
            results.groupby("decile")
            .agg(n=("label", "count"), conversions=("label", "sum"), avg_score=("score", "mean"))
            .sort_index(ascending=False)
            .assign(conversion_rate=lambda d: d["conversions"] / d["n"])
            .assign(lift=lambda d: d["conversion_rate"] / y.mean())
        )
        return decile_table

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str):
        return joblib.load(path)