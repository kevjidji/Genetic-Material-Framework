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

import chemparse
import numpy as np
import random

class GeneticAlgorithm:
    def __init__(
        self,
        elements,
        objective_model,  # MaterialModel instance
        constraint_models,  # list of dicts: {"model": MaterialModel, "min_value": float, "max_value": float}
        num_elements=3,
        min_percent=0.05,
        max_percent=0.35,
        diff_coeff=0.01,
        mutate_elements=True,
    ):
        self.elements = elements
        self.objective_model = objective_model
        self.constraint_models = constraint_models
        self.num_elements = num_elements
        self.min_percent = min_percent
        self.max_percent = max_percent
        self.diff_coeff = diff_coeff
        self.mutate_elements = mutate_elements

    def mutate(self, s, mutate_element=False, diff=0.05):
        dic = chemparse.parse_formula(s)
        if mutate_element:
            elements_choose_from = sorted(set(self.elements) - set(dic.keys()))
            if not elements_choose_from:
                return s
            replacement = random.choice(elements_choose_from)
            original = random.choice(list(dic.keys()))
            dic[replacement] = dic[original]
            del dic[original]
        else:
            more_than_min = []
            less_than_max = []
            for key in dic.keys():
                if dic[key] >= self.min_percent + diff:
                    more_than_min.append(key)
                if dic[key] <= self.max_percent - diff:
                    less_than_max.append(key)
            if not more_than_min or not less_than_max:
                return s  # fallback if mutation not possible
            take_from = random.choice(more_than_min)
            give_to = random.choice(less_than_max)
            dic[take_from] = round(dic[take_from] - diff, 3)
            dic[give_to] = round(dic[give_to] + diff, 3)

        dic = {k: dic[k] for k in sorted(dic.keys())}
        return "".join([f"{el}{dic[el]}" for el in dic.keys()])

    def safe_predict(self, model, composition):
        """
        Predict a single composition using a MaterialModel.
        Handles errors gracefully and ensures correct input format.
        """
        try:
            pred = model.predict([composition])  # predict expects a list
            if isinstance(pred, (list, np.ndarray)):
                return float(pred[0])
            return float(pred)
        except Exception as e:
            print(f"Prediction failed for {composition} with {model}: {e}")
            return -1e6  # penalize invalid predictions

    def score_population(self, population):
        """Evaluate population using constraints + objective model."""
        scores = []
        for comp in population:
            penalized = False
            for i, cm in enumerate(self.constraint_models, start=1):
                pred = self.safe_predict(cm["model"], comp)
                if pred < cm["min_value"]:
                    scores.append(-1000 * i + pred)
                    penalized = True
                    break
                elif pred > cm["max_value"]:
                    scores.append(-1000 * i - pred)
                    penalized = True
                    break
            if not penalized:
                # all constraints satisfied → use objective model
                score = self.safe_predict(self.objective_model, comp)
                scores.append(score)
        return scores

    def run(self, rounds=20, population_size=200):
        # initialize population
        population = []
        for _ in range(population_size):
            element_sample = random.sample(self.elements, self.num_elements)
            frac = round(1.0 / self.num_elements, 3)
            comp = "".join([f"{e}{frac}" for e in element_sample])
            population.append(comp)

        best_each_round = []
        top_scores = []

        for r in range(rounds):
            scores = self.score_population(population)
            inds = np.argsort(scores)
            sorted_scores = np.array(scores)[inds]
            sorted_children = np.array(population)[inds]

            # keep top 50
            passing_children = sorted_children[-50:]
            top_scores.append(sorted_scores[-50:])

            print(f"\n=== Round {r} ===")
            print("Top 5 Scores:", sorted_scores[-5:])
            print("Top 5 Compositions:", passing_children[-5:])

            best_each_round.append(passing_children[-1])

            # generate new children
            children = []
            for pc in passing_children:
                for _ in range(20):
                    children.append(self.mutate(pc, diff=self.diff_coeff * random.randint(1, 8)))
                for _ in range(10):
                    if self.mutate_elements:
                        children.append(self.mutate(pc, mutate_element=True))

            children = list(set(children + passing_children.tolist()))

            print("Children Generated:", len(children))
            population = children

            # stopping criterion: last 5 rounds same best
            if len(best_each_round) > 4:
                if all(best_each_round[-1] == x for x in best_each_round[-5:]):
                    print("Converged early.")
                    break

        return top_scores, best_each_round
