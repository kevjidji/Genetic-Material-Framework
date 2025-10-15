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




class MaterialFeaturizer:
    """
    Featurizes a list of chemical compositions into numerical descriptors
    using matminer composition-based featurizers.
    Ensures all outputs are numeric floats for ML models.
    """

    def __init__(self, featurizers=None):
        if featurizers is None:
            self.featurizers = [
                Stoichiometry(),
                ElementProperty.from_preset("magpie"),
                ValenceOrbital(),
                IonProperty(fast=True),
            ]
        else:
            self.featurizers = featurizers

    def _clean_formula(self, formula):
        """Normalize formula strings (remove -, _, and spaces)."""
        if isinstance(formula, str):
            return (
                formula.replace("-", "")
                       .replace("_", "")
                       .replace(" ", "")
                       .replace("–", "")
                       .replace("·", "")

            )
        return formula

    def featurize_list(self, compositions):
        """
        Featurize a list of composition strings into a pandas DataFrame.

        Args:
            compositions (list of str): e.g. ['Fe2O3', 'Li-Fe-PO4', ...]

        Returns:
            pd.DataFrame: numeric feature DataFrame (float dtype)
        """
        # Clean up input formulas
        clean_comps = [self._clean_formula(c) for c in compositions]

        # Convert to pymatgen Composition objects
        comp_objs = []
        for c in clean_comps:
          try:

            comp_objs.append(Composition(c))
          except:

            assert(False)

        df = pd.DataFrame({"Composition": comp_objs})

        # Apply each featurizer sequentially
        for f in self.featurizers:
            print(f"Applying featurizer: {f.__class__.__name__}")
            df = f.featurize_dataframe(df, "Composition", ignore_errors=True)

        # Drop Composition column
        df = df.drop(columns=["Composition"], errors="ignore")

        # Ensure all numeric and cast to float
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.fillna(df.mean(numeric_only=True))
        df = df.astype(float)

        return df
