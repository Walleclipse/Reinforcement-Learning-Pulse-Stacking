import os
import numpy as np

from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3

ALGOS = {
    "a2c": A2C,
    "ddpg": DDPG,
    "dqn": DQN,
    "ppo": PPO,
    "sac": SAC,
    "td3": TD3,
}

def create_action_noise(hyperparams):
    # Parse noise string
    # Note: only off-policy algorithms are supported
    if hyperparams.get("noise_type") is not None:
        noise_type = hyperparams["noise_type"].strip()
        noise_std = hyperparams["noise_std"]

        # Save for later (hyperparameter optimization)
        n_actions = hyperparams['stage']

        if "normal" in noise_type:
            hyperparams["action_noise"] = NormalActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions),
            )
        elif "ornstein-uhlenbeck" in noise_type:
            hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
                mean=np.zeros(n_actions),
                sigma=noise_std * np.ones(n_actions),
            )
        else:
            raise RuntimeError(f'Unknown noise type "{noise_type}"')

        #print(f"Applying {noise_type} noise with std {noise_std}")
    return hyperparams


def load_pretrained_agent(hyperparams=None, env=None,algo_hps=None, model_path=None):
    # Continue training
    print("Loading pretrained agent")
    # Policy should not be changed
    if model_path is not None:
        model = ALGOS[hyperparams['algo']].load(model_path)
        return model

    model = ALGOS[hyperparams['algo']].load(
        hyperparams['trained_agent'],
        env=env,
        **algo_hps,
    )

    replay_buffer_path = os.path.join(os.path.dirname(hyperparams['trained_agent']), "replay_buffer.pkl")

    if os.path.exists(replay_buffer_path):
        print("Loading replay buffer")
        # `truncate_last_traj` will be taken into account only if we use HER replay buffer
        trunc_traj = False if 'truncate_last_trajectory' not in hyperparams else hyperparams['truncate_last_trajectory']
        model.load_replay_buffer(replay_buffer_path, truncate_last_traj=trunc_traj)
    return model

def save_agent(model, hyperparams):
    model_path = os.path.join(hyperparams['model_path'], "trained_agent")
    print(f"Saving to {model_path}")
    model.save(model_path)

    #if hasattr(model, "save_replay_buffer"):
    #    print("Saving replay buffer")
    #    model.save_replay_buffer(os.path.join(hyperparams['model_path'], "replay_buffer.pkl"))

    if hyperparams['normalize']:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(os.path.join(hyperparams['env_path'], "vecnormalize.pkl"))

    return model_path

def create_model(algo,env,hyperparams):
    model = ALGOS[algo](
        env=env,
        **hyperparams,
    )
    return model
