Saving/Resuming Study with RDB Backend (https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/001_rdb.html)

An RDB backend enables persistent experiments (i.e., to save and resume a study) as well as access to history of studies. In addition, we can run multi-node optimization tasks with this feature, which is described in Easy Parallelization.

In this section, let’s try simple examples running on a local environment with SQLite DB.

Note

You can also utilize other RDB backends, e.g., PostgreSQL or MySQL, by setting the storage argument to the DB’s URL. Please refer to SQLAlchemy’s document for how to set up the URL.
New Study

We can create a persistent study by calling create_study() function as follows. An SQLite file example.db is automatically initialized with a new study record.

import logging
import sys

import optuna

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
study_name = "example-study"  # Unique identifier of the study.
storage_name = f"sqlite:///{study_name}.db"
study = optuna.create_study(study_name=study_name, storage=storage_name)

To run a study, call optimize() method passing an objective function.

def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


study.optimize(objective, n_trials=3)

Resume Study

To resume a study, instantiate a Study object passing the study name example-study and the DB URL sqlite:///example-study.db.

study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
study.optimize(objective, n_trials=3)

Note that the storage doesn’t store the state of the instance of samplers and pruners. When we resume a study with a sampler whose seed argument is specified for reproducibility, you need to restore the sampler with using pickle as follows:

import pickle

# Save the sampler with pickle to be loaded later.
with open("sampler.pkl", "wb") as fout:
    pickle.dump(study.sampler, fout)

restored_sampler = pickle.load(open("sampler.pkl", "rb"))
study = optuna.create_study(
    study_name=study_name, storage=storage_name, load_if_exists=True, sampler=restored_sampler
)
study.optimize(objective, n_trials=3)

Experimental History

Note that this section requires the installation of Pandas:

$ pip install pandas

We can access histories of studies and trials via the Study class. For example, we can get all trials of example-study as:

study = optuna.create_study(study_name=study_name, storage=storage_name, load_if_exists=True)
df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

The method trials_dataframe() returns a pandas dataframe like:

print(df)

A Study object also provides properties such as trials, best_value, best_params (see also Lightweight, versatile, and platform agnostic architecture).

print("Best params: ", study.best_params)
print("Best value: ", study.best_value)
print("Best Trial: ", study.best_trial)
print("Trials: ", study.trials)
