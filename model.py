from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import chemparse
import re
from sklearn.utils import shuffle
import lightgbm as lgb
from sklearn.impute import KNNImputer
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
import random
from pymatgen.core import Composition
from matminer.featurizers.composition import (
    ElementProperty,
    ValenceOrbital,
    IonProperty,
    Stoichiometry,
)
from sklearn.model_selection import GridSearchCV
import warnings

import json
import numpy as np
from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score
import random
from copy import deepcopy

warnings.filterwarnings("ignore")

import json
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, make_scorer
)
from sklearn.utils import shuffle
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb
from copy import deepcopy


def cross_validate(model, X, y, task, cv=5):
    if task == "regression":
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        scoring = {
            "MAE": make_scorer(mean_absolute_error, greater_is_better=False),
            "MSE": make_scorer(mean_squared_error, greater_is_better=False),
            "R2": make_scorer(r2_score),
        }
    else:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scoring = {
            "Accuracy": make_scorer(accuracy_score),
            "F1": make_scorer(f1_score, average="weighted"),
        }

    results = {}
    for metric, scorer in scoring.items():
        scores = cross_val_score(model, X, y, cv=kf, scoring=scorer)
        results[metric] = {"mean": np.mean(scores), "std": np.std(scores)}

    if task != "regression":
        results = list(results.values())[0]
        return results["mean"], results["std"]
    else:
        results = list(results.values())[2]
        return results["mean"], results["std"]


class MaterialModel:

    def __init__(self, task, model_type, data_csv, featurizer, hyperparams=None, ga_json_path=None):
        """
        Args:
            task (str): "regression" or "classification"
            model_type (str): "RF", "GB", "LGBM", "XGB"
            data_csv (str): path to CSV with 'Composition' + target column (+ optional feature columns)
            featurizer (MaterialFeaturizer): featurizer object
            hyperparams (dict): hyperparameters for chosen model
            ga_json_path (str, optional): path to GA optimization result JSON
        """
        self.task = task
        self.model_type = model_type
        self.data_csv = data_csv
        self.featurizer = featurizer
        self.hyperparams = hyperparams if hyperparams else {}
        self.model = None
        self.target_name = None
        self.selected_features = None
        self.csv_column_means = {}

        df = pd.read_csv(self.data_csv)

        if "Composition" not in df.columns:
            raise ValueError("CSV must contain a 'Composition' column.")

        non_comp_cols = [c for c in df.columns if c != "Composition"]
        if len(non_comp_cols) == 0:
            raise ValueError("CSV must have a target column.")
        self.target_name = non_comp_cols[0]

        # Extra numeric columns
        self.extra_features = None
        if len(non_comp_cols) > 1:
            self.extra_features = df[non_comp_cols[1:]].copy()

        # Store CSV column means (for use during predict)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.csv_column_means = df[numeric_cols].mean().to_dict()

        self.data = df

        # Featurize compositions
        X, y = self._featurize(df)
        X, y = shuffle(X, y, random_state=42)

        # If GA JSON provided, apply selected features + hyperparams
        if ga_json_path is not None:
            with open(ga_json_path, "r") as f:
                ga_data = json.load(f)
            self.hyperparams.update(ga_data.get("hyperparameters", {}))
            self.selected_features = ga_data.get("selected_features", None)

            if self.selected_features is not None:
                keep_cols = [col for col in X.columns if col in self.selected_features or col in self.extra_features.columns]
                X = X[keep_cols]

        self.X, self.y = X, y
        print(f"Featurized {len(self.X)} compositions with {self.X.shape[1]} features.")

        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.y_train_pred = self.y_test_pred = None

        # Select model
        if self.model_type == "RF":
            self.model = RandomForestRegressor(**self.hyperparams) if self.task == "regression" else RandomForestClassifier(**self.hyperparams)
        elif self.model_type == "GB":
            self.model = GradientBoostingRegressor(**self.hyperparams) if self.task == "regression" else GradientBoostingClassifier(**self.hyperparams)
        elif self.model_type == "LGBM":
            self.model = lgb.LGBMRegressor(**self.hyperparams) if self.task == "regression" else lgb.LGBMClassifier(**self.hyperparams)
        elif self.model_type == "XGB":
            self.model = XGBRegressor(**self.hyperparams) if self.task == "regression" else XGBClassifier(**self.hyperparams)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _sanitize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        safe_cols = []
        for col in df.columns:
            safe = re.sub(r'[^0-9a-zA-Z_]', '_', col)
            safe_cols.append(safe)
        df.columns = safe_cols
        return df

    def _featurize(self, df):
        X = self.featurizer.featurize_list(df["Composition"].tolist())
        X = self._sanitize_columns(X)

        if self.extra_features is not None:
            numeric_extra = self.extra_features.select_dtypes(include=[np.number])
            non_numeric = [c for c in self.extra_features.columns if c not in numeric_extra.columns]
            if non_numeric:
                print(f"Ignoring non-numeric extra columns: {non_numeric}")
            X = pd.concat([X, numeric_extra.reset_index(drop=True)], axis=1)

        imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

        y = df[self.target_name].values
        return X_imputed, y

    def predict(self, composition: str):
        """Predict a single composition using mean CSV values for auxiliary columns."""
        X_feat = self.featurizer.featurize_list([composition])
        X_feat = self._sanitize_columns(X_feat)

        # Add CSV columns with mean values
        for col, mean_val in self.csv_column_means.items():
            X_feat[col] = mean_val

        # Restrict to GA-selected features if any
        if self.selected_features is not None:
            X_feat = X_feat[[c for c in X_feat.columns if c in self.selected_features or c in self.csv_column_means]]

        features = X_feat[self.X.columns].values

        if self.task == "regression":
            return float(self.model.predict(features)[0])
        else:
            proba = self.model.predict_proba(features)[0]
            return float(proba[1]) if proba.shape[0] > 1 else float(proba[0])

    def plot_confusion_matrix(self, title="Confusion Matrix", xlabel="Predicted", ylabel="True"):
        if self.task != "classification":
            raise RuntimeError("Confusion matrix is only available for classification tasks.")

        from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

        y_pred = self.model.predict(self.X_test)
        if y_pred.ndim > 1 and y_pred.shape[1] > 1:
            y_pred = np.argmax(y_pred, axis=1)

        cm = confusion_matrix(self.y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.show()

        print("\nClassification Report:\n")
        print(classification_report(self.y_test, y_pred, digits=3))

    def plot_importances(self, top_n=10, title="Top Feature Importances", xlabel="Importance", ylabel="Features", override_labels=None):
        if not hasattr(self.model, "feature_importances_"):
            raise RuntimeError("This model does not support feature importances.")

        importances = self.model.feature_importances_
        feature_names = np.array(self.X.columns)

        sorted_idx = np.argsort(importances)[::-1][:top_n]
        top_features = feature_names[sorted_idx]
        top_importances = importances[sorted_idx]

        if override_labels is not None and len(override_labels) == top_n:
            top_features = override_labels

        plt.figure(figsize=(8, 5))
        plt.barh(range(top_n), top_importances[::-1])
        plt.yticks(range(top_n), top_features[::-1])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.show()

        print("Top features:")
        for f, imp in zip(top_features, top_importances):
            print(f"{f:25s}: {imp:.4f}")

    def plot_results(self, plot_train=True, title="Predicted vs True", xlabel="True", ylabel="Predicted"):
        if self.task != "regression":
            raise RuntimeError("plot_results is only for regression tasks.")

        if plot_train:
            y_true, y_pred = self.y_train, self.y_train_pred
            label = "Train"
        else:
            y_true, y_pred = self.y_test, self.y_test_pred
            label = "Test"

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        plt.figure(figsize=(6, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], "r--")
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title} ({label} Set)")

        textstr = f"MAE = {mae:.3f}\nMSE = {mse:.3f}\nR² = {r2:.3f}"
        plt.gca().text(
            0.95, 0.05, textstr,
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8)
        )
        plt.show()

    def cross_validate(self, cv=5):
        return cross_validate(self.model, self.X, self.y, self.task, cv=cv)
