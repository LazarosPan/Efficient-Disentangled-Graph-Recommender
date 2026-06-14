Callback for Study.optimize (https://optuna.readthedocs.io/en/stable/tutorial/20_recipes/007_optuna_callback.html)

This tutorial showcases how to use & implement Optuna Callback for optimize().

Callback is called after every evaluation of objective, and it takes Study and FrozenTrial as arguments, and does some work.

MLflowCallback is a great example.
Stop optimization after some trials are pruned in a row

This example implements a stateful callback which stops the optimization if a certain number of trials are pruned in a row. The number of trials pruned in a row is specified by threshold.
```python
import optuna

class StopWhenTrialKeepBeingPrunedCallback:
    def __init__(self, threshold: int):
        self.threshold = threshold
        self._consequtive_pruned_count = 0

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.state == optuna.trial.TrialState.PRUNED:
            self._consequtive_pruned_count += 1
        else:
            self._consequtive_pruned_count = 0

        if self._consequtive_pruned_count >= self.threshold:
            study.stop()
```
This objective prunes all the trials except for the first 5 trials (trial.number starts with 0).

def objective(trial):
    if trial.number > 4:
        raise optuna.TrialPruned

    return trial.suggest_float("x", 0, 1)

Here, we set the threshold to 2: optimization finishes once two trials are pruned in a row. So, we expect this study to stop after 7 trials.

import logging
import sys

# Add stream handler of stdout to show the messages
optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

study_stop_cb = StopWhenTrialKeepBeingPrunedCallback(2)
study = optuna.create_study()
study.optimize(objective, n_trials=10, callbacks=[study_stop_cb])
