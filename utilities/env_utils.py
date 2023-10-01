import os
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


def create_envs(hyperparams,env_hyperparams, eval_env=False, no_log=False):
    """
    Create the environment and wrap it if necessary.

    :param n_envs:
    :param eval_env: Whether is it an environment used for evaluation or not
    :param no_log: Do not log training when doing hyperparameter optim
        (issue with writing the same file)
    :return: the vectorized environment, with appropriate wrappers
    """
    # Do not log eval env (issue with writing the same file)
    log_dir = None if eval_env or no_log else hyperparams['env_path']

    # On most env, SubprocVecEnv does not help and is quite memory hungry
    # therefore we use DummyVecEnv by default
    env_kwargs = env_hyperparams.copy()
    vec_env_kwargs = hyperparams['vec_env_kwargs'].copy()
    if eval_env:
        env_kwargs['max_episode_steps'] = min(200, env_kwargs['max_episode_steps'])
    env = make_vec_env(
        env_id=hyperparams['env_id'],
        n_envs=1,
        seed=env_kwargs['seed'],
        env_kwargs=env_kwargs,
        vec_env_cls=hyperparams['vec_env_class'],
        vec_env_kwargs=vec_env_kwargs,
        monitor_dir=log_dir,
    )

    # Wrap the env into a VecNormalize wrapper if needed
    # and load saved statistics when present
    env = _maybe_normalize(hyperparams, env, eval_env)

    # Optional Frame-stacking
    if hyperparams['frame_stack'] is not None:
        n_stack = hyperparams['frame_stack']
        env = VecFrameStack(env, n_stack)
        if hyperparams['verbose'] > 0:
            print(f"Stacking {n_stack} frames")
    return env

def _maybe_normalize(hyperparams, env, eval_env):
    """
    Wrap the env into a VecNormalize wrapper if needed
    and load saved statistics when present.

    :param env:
    :param eval_env:
    :return:
    """
    # Pretrained model, load normalization
    path_ = os.path.join(hyperparams['env_path'], "vecnormalize.pkl")

    if os.path.exists(path_):
        print("Loading saved VecNormalize stats")
        env = VecNormalize.load(path_, env)
        # Deactivate training and reward normalization
        if eval_env:
            env.training = False
            env.norm_reward = False

    elif hyperparams['normalize']:
        # Copy to avoid changing default values by reference
        local_normalize_kwargs = hyperparams['normalize_kwargs'].copy()
        # Do not normalize reward for env used for evaluation
        if eval_env:
            if len(local_normalize_kwargs) > 0:
                local_normalize_kwargs["norm_reward"] = False
            else:
                local_normalize_kwargs = {"norm_reward": False}

        if hyperparams['verbose'] > 0:
            if len(local_normalize_kwargs) > 0:
                print(f"Normalization activated: {local_normalize_kwargs}")
            else:
                print("Normalizing input and reward")
        env = VecNormalize(env, **local_normalize_kwargs)
    return env
