import os

import yaml
import argparse
from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from .spgd_vec_env import SPGDVecEnv

def linear_schedule(initial_value):
    """
    Linear learning rate schedule.

    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func


def init_trial_path(out_dir):
    """Initialize the path for a hyperparameter setting
    """
    os.makedirs(out_dir, exist_ok=True)
    trial_id = 0
    path_exists = True
    path_to_results = out_dir + '/{:d}'.format(trial_id)
    while path_exists:
        trial_id += 1
        path_to_results = out_dir + '/{:d}'.format(trial_id)
        path_exists = os.path.exists(path_to_results)
    save_path = path_to_results
    os.makedirs(save_path, exist_ok=True)
    return save_path


def read_hyperparams(args):
    hyperparams = {}
    if os.path.exists(args.default_hp_path):
        with open(args.default_hp_path, "r") as f:
            hyperparams_dict = yaml.safe_load(f)
            if 'env' in list(hyperparams_dict.keys()):
                hyperparams.update(hyperparams_dict['env'])
            if 'algo' in list(hyperparams_dict.keys()):
                hyperparams.update(hyperparams_dict['algo'])
            if args.algo in list(hyperparams_dict.keys()):
                hyperparams.update(hyperparams_dict[args.algo])
    else:
        print(f"Hyperparameters not found for {args.default_hp_path}")

    custom_hyperparams = args.__dict__
    if 'hyperparams' in custom_hyperparams:
        over_hp = custom_hyperparams['hyperparams']
        if over_hp is not None and len(over_hp)>0:
            for k,v in over_hp.items():
                custom_hyperparams[k]=v
        del custom_hyperparams['hyperparams']

    hyperparams.update(custom_hyperparams)
    return hyperparams


def _preprocess_schedules(hyperparams):
    for key in ["learning_rate", "clip_range", "clip_range_vf"]:
        if key not in hyperparams:
            continue
        if isinstance(hyperparams[key], str):
            schedule, initial_value = hyperparams[key].split("_")
            initial_value = float(initial_value)
            hyperparams[key] = linear_schedule(initial_value)
        elif isinstance(hyperparams[key], (float, int)):
            # Negative value: ignore (ex: for clipping)
            continue
            if hyperparams[key] < 0:
                continue
            hyperparams[key] = constant_fn(float(hyperparams[key]))
        else:
            raise ValueError(f"Invalid value for {key}: {hyperparams[key]}")
    return hyperparams


def _preprocess_normalization(hyperparams):
    if "normalize" in hyperparams.keys():
        normalize = hyperparams["normalize"]

        # Special case, instead of both normalizing
        # both observation and reward, we can normalize one of the two.
        # in that case `hyperparams["normalize"]` is a string
        # that can be evaluated as python,
        # ex: "dict(norm_obs=False, norm_reward=True)"
        if isinstance(normalize, str):
            hyperparams['normalize_kwargs'] = eval(normalize)
            hyperparams['normalize'] = True
        if 'normalize_kwargs' not in hyperparams:
            hyperparams['normalize_kwargs'] = {}
        # Use the same discount factor as for the algorithm
        if "gamma" in hyperparams:
            hyperparams['normalize_kwargs']["gamma"] = hyperparams["gamma"]
    else:
        hyperparams['normalize'] = False
    if 'normalize_kwargs' not in hyperparams:
        hyperparams['normalize_kwargs']={}
    return hyperparams

def preprocess_folder(hyperparams):
    # init trial path
    result_dir = os.path.join(hyperparams['out_dir'], hyperparams['env_name'], hyperparams['algo'])
    hyperparams['log_folder'] = init_trial_path(result_dir)
    hyperparams['log_path'] = os.path.join(hyperparams['log_folder'],'log')
    hyperparams['env_path'] = os.path.join(hyperparams['log_folder'],'env')
    hyperparams['model_path'] = os.path.join(hyperparams['log_folder'], 'ckpt')
    os.makedirs(hyperparams['log_path'],exist_ok=True)
    os.makedirs(hyperparams['model_path'], exist_ok=True)
    os.makedirs(hyperparams['env_path'], exist_ok=True)

    if 'tensorboard_log' in hyperparams:
        tb = hyperparams['tensorboard_log']
        if tb is not None and len(tb)>0:
            hyperparams['tensorboard_log'] =  os.path.join(hyperparams['log_folder'],'tensorboard')
            os.makedirs(hyperparams['tensorboard_log'], exist_ok=True)
    else:
        hyperparams['tensorboard_log'] = None

    return hyperparams

def process_hyperparams(args):
    # Load hyperparameters from yaml file
    hyperparams = read_hyperparams(args)
    hyperparams['env_name'] =hyperparams['env_id'].__name__ + '_'+ str(hyperparams['stage'])

    # Convert schedule strings to objects
    hyperparams = _preprocess_schedules(hyperparams)
    # Pre-process normalize config
    hyperparams = _preprocess_normalization(hyperparams)
    # Pre-process folder
    hyperparams = preprocess_folder(hyperparams)

    # Pre-process train_freq
    if "train_freq" in hyperparams and isinstance(hyperparams["train_freq"], list):
        hyperparams["train_freq"] = tuple(hyperparams["train_freq"])

    # Pre-process policy/buffer keyword arguments
    # Convert to python object if needed
    for kwargs_key in {"policy_kwargs", "replay_buffer_class", "replay_buffer_kwargs"}:
        if kwargs_key in hyperparams.keys() and isinstance(hyperparams[kwargs_key], str):
            hyperparams[kwargs_key] = eval(hyperparams[kwargs_key])

    if hyperparams['algo']=='ppo' and 'tau' in hyperparams:
        del hyperparams['tau']
    return hyperparams


ENV_HP_KEYS = ['stage', 'noise_sigma', 'max_episode_steps','obs_step','obs_noise_sigma', 'init_nonoptimal', 'action_scale', 'obs_signal',
               'normalized_action', 'normalized_observation', 'max_pzm', 'max_episode_steps', 'reward_threshold',
               'spgd', 'perturb_scale', 'dict_observation', 'noise_loc', 'init_loc', 'noise_evolve','difficulty']

SKIP_KEYS = ['algo','n_timesteps', 'frame_stack', 'noise_type', 'noise_std', 'normalize','normalize_kwargs',
             'out_dir','log_folder','log_path','model_path','env_path','trained_agent',
             'default_hp_path','eval_freq','n_eval_episodes','save_freq','env_id','env_name',
             'optimize_hyperparameters','deterministic_eval','vec_env',]


def split_hyperparams(hyperparams,is_save=True):
    algo_hyperparams = {}
    env_hyperparams = {}
    exp_hyperparams = {}
    spgd_hyperparameters = {}
    for k, v in hyperparams.items():
        if 'spgd_' in k:
            spgd_hyperparameters[k]=v
        elif k in ENV_HP_KEYS:
            env_hyperparams[k] = v
        elif k not in SKIP_KEYS:
            algo_hyperparams[k] = v
        else:
            exp_hyperparams[k] = v

    env_hyperparams['seed'] = hyperparams['seed']
    algo_hyperparams['seed'] = hyperparams['seed']
    exp_hyperparams['seed'] = hyperparams['seed']
    if 'verbose' not in hyperparams:
        hyperparams['verbose']=0
    exp_hyperparams['verbose'] = hyperparams['verbose']

    if hyperparams['vec_env']=='dummy':
        exp_hyperparams['vec_env_class'] = DummyVecEnv
        exp_hyperparams['vec_env_kwargs'] ={}
    elif hyperparams['vec_env']=='subproc':
        exp_hyperparams['vec_env_class'] = SubprocVecEnv
        exp_hyperparams['vec_env_kwargs'] ={"start_method": "fork"}
    elif hyperparams['vec_env']=='spgd':
        exp_hyperparams['vec_env_class'] = SPGDVecEnv
        exp_hyperparams['vec_env_kwargs'] =spgd_hyperparameters
    else:
        exp_hyperparams['vec_env_class'] = None
        exp_hyperparams['vec_env_kwargs'] ={}

    if is_save:
        save_hyperparams(algo_hyperparams, env_hyperparams, exp_hyperparams, spgd_hyperparameters)
    return algo_hyperparams, env_hyperparams, exp_hyperparams, spgd_hyperparameters

def filter_value(v):
    if isinstance(v, (int, float, str, bool)):
        return v
    if isinstance(v, (list, tuple)):
        if len(v)==0:
            return v
        else:
            new_v=[]
            for vv in v:
                new_v.append(filter_value(vv))
            return new_v
    if isinstance(v, dict):
        if len(v)==0:
            return v
        else:
            new_v={}
            for vk, vv in v.items():
                new_v[vk] = filter_value(vv)
            return new_v
    return None

def filter_hyperparames(hps):
    new_hps={}
    for k,v in hps.items():
        new_v = filter_value(v)
        if new_v is not None:
            new_hps[k]=new_v
    return new_hps

def save_hyperparams(algo_hyperparams, env_hyperparams, exp_hyperparams, spgd_hyperparameters):
    saved_params={'env':{},'algo':{},'exp':{},'spgd':{}}

    saved_params['env'] = filter_hyperparames(env_hyperparams)
    saved_params['algo'] = filter_hyperparames(algo_hyperparams)
    saved_params['exp'] = filter_hyperparames(exp_hyperparams)
    saved_params['spgd'] = filter_hyperparames(spgd_hyperparameters)

    if exp_hyperparams['verbose']:
        print('Saved parameters:')
        print('env:',saved_params['env'])
        print('algo:',saved_params['algo'])
        print('exp:',saved_params['exp'])
        print('spgd:',saved_params['spgd'])

    with open(os.path.join(exp_hyperparams['log_folder'], "hyperparams.yml"), "w") as f:
        yaml.dump(saved_params, f)

class StoreDict(argparse.Action):
    """
    Custom argparse action for storing dict.

    In: args1:0.0 args2:"dict(a=1)"
    Out: {'args1': 0.0, arg2: dict(a=1)}
    """

    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDict, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        arg_dict = {}
        for arguments in values:
            key = arguments.split(":")[0]
            value = ":".join(arguments.split(":")[1:])
            # Evaluate the string as python code
            arg_dict[key] = eval(value)
        setattr(namespace, self.dest, arg_dict)