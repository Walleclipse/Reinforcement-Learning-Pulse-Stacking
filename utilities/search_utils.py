import os
import warnings
import optuna
from optuna.integration.skopt import SkoptSampler
from optuna.pruners import BasePruner, MedianPruner, SuccessiveHalvingPruner
from optuna.samplers import BaseSampler, RandomSampler, TPESampler




def create_sampler(sampler_method,n_startup_trials, seed):
    # n_warmup_steps: Disable pruner until the trial reaches the given number of step.
    if sampler_method == "random":
        sampler = RandomSampler(seed=seed)
    elif sampler_method == "tpe":
        # TODO: try with multivariate=True
        sampler = TPESampler(n_startup_trials=n_startup_trials, seed=seed)
    elif sampler_method == "skopt":
        # cf https://scikit-optimize.github.io/#skopt.Optimizer
        # GP: gaussian process
        # Gradient boosted regression: GBRT
        sampler = SkoptSampler(skopt_kwargs={"base_estimator": "GP", "acq_func": "gp_hedge"})
    else:
        raise ValueError(f"Unknown sampler: {sampler_method}")
    return sampler


def create_pruner(pruner_method,n_startup_trials,n_evaluations,n_trials) -> BasePruner:
    if pruner_method == "halving":
        pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=4, min_early_stopping_rate=0)
    elif pruner_method == "median":
        pruner = MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_evaluations // 3)
    elif pruner_method == "none":
        # Do not prune
        pruner = MedianPruner(n_startup_trials=n_trials, n_warmup_steps=n_evaluations)
    else:
        raise ValueError(f"Unknown pruner: {pruner_method}")
    return pruner

