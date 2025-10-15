from featurizer import MaterialFeaturizer
from model import MaterialModel
from genetic_algorithm import GeneticAlgorithm

phase_csv = "phase_prediction.csv"     # training CSV for phase classifier
delta_h_csv = "delta_h.csv"
solid_solution_csv = "weight_percent.csv"
featurizer = MaterialFeaturizer()

"""
Initialize our models.

OPTIONAL: pass in the parameter ga_json_path for the json of hyperparameters and features obtained using example_hyperparameter_optimization.py

"""
model_phase = MaterialModel("classification", "RF", phase_csv, featurizer,predicted_class="BCC")
model_solid_solution = MaterialModel("regression","RF",solid_solution_csv,featurizer)
model_delta_H = MaterialModel("regression","RF",delta_h_csv,featurizer)

model_phase.train()
model_delta_H.train()
model_solid_solution.train()

genetic_algorithm = GeneticAlgorithm(["Mg", "Al", "Ti", "V", "Cr", "Mn",
"Fe", "Co", "Ni", "Cu", "Nb",  "Mo"],model_solid_solution,[{"model": model_phase, "min_value": 0.9, "max_value": 1.0},{"model": model_delta_H, "min_value": 40, "max_value": 200}],5)

results = genetic_algorithm.run(rounds=20,population_size=200)