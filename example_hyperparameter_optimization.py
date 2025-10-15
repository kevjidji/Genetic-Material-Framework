from featurizer import MaterialFeaturizer
from model import MaterialModel
from genetic_algorithm import GeneticAlgorithm


phase_csv = "phase_prediction.csv"     # training CSV for phase classifie
featurizer = MaterialFeaturizer()

model_phase = MaterialModel("classification", "RF", phase_csv, featurizer,predicted_class="BCC")

model_phase.genetic_algorithm_optimize(param_space = {
    "n_estimators": (200, 600, int),       
    "max_depth": (3, 20, int),             
    "min_samples_split": (2, 15, int),    
    "min_samples_leaf": (1, 8, int),
    "max_features": (0.3, 0.9, float),     
    "max_samples": (0.7, 1.0, float)       
}
, population_size=15,feature_fraction_default=0.9)