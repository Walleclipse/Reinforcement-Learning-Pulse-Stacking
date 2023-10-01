from typing import Any, Dict

import numpy as np
import optuna
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch import nn as nn

from .hp_utils import linear_schedule

def sample_spgd_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SPGD hyperparams.

    :param trial:
    :return:
    """
    spgd_factor = trial.suggest_categorical("spgd_factor", [0, 0.1, 0.3, 0.5])
    spgd_factor_test = trial.suggest_categorical("spgd_factor_test",[0, 0.1, 0.3, 0.5])
    spgd_begin = trial.suggest_categorical("spgd_begin", [1, 10, 100])
    spgd_warmup = trial.suggest_categorical("spgd_warmup", [1, 10, 100])
    spgd_lr = trial.suggest_categorical("spgd_lr", [1, 10, 100, 1000, 10000])
    spgd_momentum = trial.suggest_categorical("spgd_momentum", [0, 0.1, 0.3, 0.5])
    hyperparams = {
        "spgd_factor": spgd_factor,
        "spgd_factor_test":spgd_factor_test,
        "spgd_begin": spgd_begin,
        "spgd_warmup": spgd_warmup,
        "spgd_lr": spgd_lr,
        "spgd_momentum": spgd_momentum,
    }

    return hyperparams

def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [ 64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[128, 128], vf=[128, 128])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
        "big": [dict(pi=[400, 300], vf=[400, 300])],
        "verybig":[dict(pi=[256, 256,256], vf=[256, 256,256])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
        ),
    }


def sample_ppo_params_old(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    #batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
    n_steps = trial.suggest_categorical("n_steps", [256, 512, 1024, 2048])
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.999])
    learning_rate = trial.suggest_categorical("lr", [3e-4,1e-3,3e-3])
    lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    ent_coef = trial.suggest_categorical("ent_coef", [0, 0.001, 0.01])
    #ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 10, 50])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.95,0.98, 0.99])
    #max_grad_norm = trial.suggest_categorical("max_grad_norm", [ 0.5, 1, 5])
    max_grad_norm = 0.5
    vf_coef = trial.suggest_uniform("vf_coef", 0.1, 0.9)
    net_arch = trial.suggest_categorical("net_arch", ["small","medium", "verybig"])
    #log_std_init=-2
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    use_sde = trial.suggest_categorical("use_sde", [True, False])
    #use_sde = True
    if use_sde:
        sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 16])
    else:
        sde_sample_freq = -1
    # Orthogonal initialization
    #ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    #activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    activation_fn = 'relu'
    # TODO: account when using multiple envs
    #if batch_size > n_steps:
    #    batch_size = n_steps

    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "small": [dict(pi=[128, 128], vf=[128, 128])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
        "big": [dict(pi=[400, 300], vf=[400, 300])],
        "verybig":[dict(pi=[256, 256,256], vf=[256, 256,256])],
    }[net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    hyperparams = {
        "n_steps": n_steps,
        #"batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "use_sde":use_sde,
        "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            #log_std_init=log_std_init,
            net_arch=net_arch,
            #activation_fn=activation_fn,
            #ortho_init=ortho_init,
        ),
    }


    return hyperparams


def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.99, 0.999])
    learning_rate = trial.suggest_loguniform("lr", 5e-5, 1e-2)
    #lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    lr_schedule = "constant"
    #gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
    #max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 1])
    max_grad_norm = 0.5
    # Toggle PyTorch RMS Prop (different from TF one, cf doc)
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
    #use_rms_prop = True
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9,  0.95,  1.0])
    n_steps = trial.suggest_categorical("n_steps", [4, 8, 16, 64])

    #ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
    ent_coef = trial.suggest_categorical("ent_coef", [0, 0.001, 0.01])
    vf_coef = trial.suggest_uniform("vf_coef", 0.1, 0.9)
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
    #log_std_init=-2
    #ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    #ortho_init = False
    net_arch = trial.suggest_categorical("net_arch", ["medium", "big","verybig"])
    # sde_net_arch = trial.suggest_categorical("sde_net_arch", [None, "tiny", "small"])
    # full_std = trial.suggest_categorical("full_std", [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    #activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    use_sde = trial.suggest_categorical("use_sde", [True, False])
    if use_sde:
        sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1,4, 16])
    else:
        sde_sample_freq = -1

    activation_fn ='relu'
    if lr_schedule == "linear":
        learning_rate = linear_schedule(learning_rate)

    net_arch = {
        "small": [dict(pi=[64, 64], vf=[64, 64])],
        "medium": [dict(pi=[256, 256], vf=[256, 256])],
        "big":[dict(pi=[400, 300], vf=[400, 300])],
        "verybig": [dict(pi=[256, 256, 256], vf=[256, 256, 256])],
    }[net_arch]

    # sde_net_arch = {
    #     None: None,
    #     "tiny": [64],
    #     "small": [64, 64],
    # }[sde_net_arch]

    activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

    hyperparams={
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "normalize_advantage": normalize_advantage,
        "max_grad_norm": max_grad_norm,
        "use_rms_prop": use_rms_prop,
        "vf_coef": vf_coef,
        "use_sde":use_sde,
        "sde_sample_freq":sde_sample_freq,
        "policy_kwargs": dict(
            #log_std_init=log_std_init,
            net_arch=net_arch,
            # full_std=full_std,
            #activation_fn=activation_fn,
            # sde_net_arch=sde_net_arch,
            #ortho_init=ortho_init,
        ),
    }


    return hyperparams

def sample_sac_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for SAC hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.999])
    learning_rate = trial.suggest_categorical("lr", [3e-4,1e-3,3e-3])
    #learning_rate = trial.suggest_loguniform("lr", 2e-4, 5e-3)
    #batch_size = trial.suggest_categorical("batch_size", [128, 256, 512,])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    #learning_starts = trial.suggest_categorical("learning_starts", [100, 1000])
    #train_freq = trial.suggest_categorical('train_freq', [1, 10])
    #train_freq = trial.suggest_categorical("train_freq", [8, 16, 32, 64, 128, 256, 512])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.002, 0.005, 0.01])
    # gradient_steps takes too much time
    # gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    #gradient_steps = train_freq
    #ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])
    ent_coef = "auto"
    # You can comment that out when not using gSDE
    #log_std_init = trial.suggest_uniform("log_std_init", -4, -1)
    log_std_init=-2
    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch = trial.suggest_categorical("net_arch", ["small", "medium","verybig"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])
    use_sde = True # trial.suggest_categorical("use_sde", [True, False])
    if use_sde:
        sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1,4, 16, 64])
    else:
        sde_sample_freq = -1

    net_arch = {
        "small": [128, 128],
        "medium": [256, 256],
        "big": [400, 300],
        "verybig": [256, 256, 256],
    }[net_arch]

    target_entropy = "auto"
    # if ent_coef == 'auto':
    #     # target_entropy = trial.suggest_categorical('target_entropy', ['auto', 5, 1, 0, -1, -5, -10, -20, -50])
    #     target_entropy = trial.suggest_uniform('target_entropy', -10, 10)

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        #"batch_size": batch_size,
        "buffer_size": buffer_size,
        #"learning_starts": learning_starts,
        #"train_freq": train_freq,
        #"gradient_steps": gradient_steps,
        "ent_coef": ent_coef,
        "tau": tau,
        "target_entropy": target_entropy,
        "use_sde": use_sde,
        "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(log_std_init=log_std_init, net_arch=net_arch),
    }

    #if trial.using_her_replay_buffer:
    #    hyperparams = sample_her_params(trial, hyperparams)


    return hyperparams


def sample_td3_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for TD3 hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.999])
    learning_rate = trial.suggest_categorical("lr", [3e-4,1e-3,3e-3])
    #batch_size = trial.suggest_categorical("batch_size", [128, 256, 512,])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    episodic = trial.suggest_categorical("episodic", [True, False])
    #episodic = False

    if episodic:
        train_freq, gradient_steps = (1, "episode"), -1
    else:
        train_freq = trial.suggest_categorical("train_freq", [1, 10, 100])
        gradient_steps = train_freq

    #noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_type = "normal"
    noise_std = trial.suggest_uniform("noise_std", 0.1, 0.9)

    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch = trial.suggest_categorical("net_arch", [ "small","medium",  "verybig"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [128, 128],
        "medium": [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        "verybig": [256, 256, 256],
    }[net_arch]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        #"batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),

    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    #if trial.using_her_replay_buffer:
    #    hyperparams = sample_her_params(trial, hyperparams)

    return hyperparams


def sample_ddpg_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for DDPG hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.98, 0.99, 0.999])
    learning_rate = trial.suggest_loguniform("lr", 2e-4, 5e-3)
    batch_size = trial.suggest_categorical("batch_size", [128, 256, 512,])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.005, 0.01])

    #episodic = trial.suggest_categorical("episodic", [True, False])
    episodic = False
    if episodic:
        train_freq, gradient_steps = (1, "episode"), -1
    else:
        train_freq = trial.suggest_categorical("train_freq", [1, 10])
        gradient_steps = train_freq

    #noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_type = "normal"
    noise_std = trial.suggest_uniform("noise_std", 0.1, 0.9)

    # NOTE: Add "verybig" to net_arch when tuning HER (see TD3)
    net_arch = trial.suggest_categorical("net_arch", ["medium", "big", "verybig"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        "verybig": [256, 256, 256],
    }[net_arch]

    hyperparams = {
        "gamma": gamma,
        "tau": tau,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
    }

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(trial.n_actions), sigma=noise_std * np.ones(trial.n_actions)
        )

    #if trial.using_her_replay_buffer:
    #    hyperparams = sample_her_params(trial, hyperparams)


    return hyperparams


def sample_her_params(trial: optuna.Trial, hyperparams: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sampler for HerReplayBuffer hyperparams.

    :param trial:
    :parma hyperparams:
    :return:
    """
    her_kwargs = trial.her_kwargs.copy()
    her_kwargs["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
    her_kwargs["goal_selection_strategy"] = trial.suggest_categorical(
        "goal_selection_strategy", ["final", "episode", "future"]
    )
    her_kwargs["online_sampling"] = trial.suggest_categorical("online_sampling", [True, False])
    hyperparams["replay_buffer_kwargs"] = her_kwargs
    return hyperparams


HYPERPARAMS_SAMPLER = {
    "a2c": sample_a2c_params,
    "ddpg": sample_ddpg_params,
    "sac": sample_sac_params,
    "ppo": sample_ppo_params,
    "td3": sample_td3_params,
    "spgd": sample_spgd_params,
}
