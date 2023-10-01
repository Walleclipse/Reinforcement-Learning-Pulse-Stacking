import os, sys
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import HumanOutputFormat, CSVOutputFormat, TensorBoardOutputFormat, Logger
from .env_utils import create_envs

class SaveVecNormalizeCallback(BaseCallback):
    """
    Callback for saving a VecNormalize wrapper every ``save_freq`` steps

    :param save_freq: (int)
    :param save_path: (str) Path to the folder where ``VecNormalize`` will be saved, as ``vecnormalize.pkl``
    :param name_prefix: (str) Common prefix to the saved ``VecNormalize``, if None (default)
        only one file will be kept.
    """

    def __init__(self, save_freq, save_path, name_prefix = None, verbose = 0):
        super(SaveVecNormalizeCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if self.name_prefix is not None:
                path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            else:
                path = os.path.join(self.save_path, "vecnormalize.pkl")
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(path)
                if self.verbose > 1:
                    print(f"Saving VecNormalize to {path}")
        return True


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.
    """

    def __init__(
        self,
        eval_env,
        trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 0,
    ):

        super(TrialEvalCallback, self).__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super(TrialEvalCallback, self)._on_step()
            self.eval_idx += 1
            # report best or report current ?
            # report num_timesteps or elasped time ?
            reward = self.reward_per_step
            self.trial.report(reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True



def create_callback(hyperparams,env_hyperparams=None):
    callbacks=[]
    if hyperparams['save_freq'] > 0:
        # Account for the number of parallel environments
        callbacks.append(
            CheckpointCallback(
                save_freq=hyperparams['save_freq'],
                save_path=hyperparams['model_path'],
                name_prefix="rl_model",
                verbose=hyperparams['verbose'],
            )
        )

    # Create test env if needed, do not normalize reward
    if hyperparams['eval_freq'] > 0:
        # Account for the number of parallel environment

        if hyperparams['verbose'] > 0:
            print("Creating test environment")

        save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=hyperparams['env_path'],verbose=hyperparams['verbose'])
        eval_callback = EvalCallback(
            create_envs(hyperparams,env_hyperparams, eval_env=True),
            callback_on_new_best=save_vec_normalize,
            best_model_save_path=hyperparams['model_path'],
            n_eval_episodes=hyperparams['n_eval_episodes'],
            log_path=hyperparams['log_path'],
            eval_freq=hyperparams['eval_freq'],
            deterministic=hyperparams['deterministic_eval'],
            verbose=hyperparams['verbose'],
        )

        callbacks.append(eval_callback)
    return callbacks


def create_logger(hyperparams):
    ho_log = HumanOutputFormat(sys.stdout)
    csv_log = CSVOutputFormat(os.path.join(hyperparams['log_path'],'results.csv'))
    output_formats=[ho_log,csv_log]
    if 'tensorboard_log' in hyperparams:
        tb = hyperparams['tensorboard_log']
        if tb is not None and len(tb) > 0:
            tb_log = TensorBoardOutputFormat(hyperparams['tensorboard_log'])
            output_formats.append(tb_log)
    logger = Logger(folder=None, output_formats=output_formats)
    return logger
