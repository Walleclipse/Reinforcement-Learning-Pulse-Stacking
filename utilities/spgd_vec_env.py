from collections import OrderedDict
from copy import deepcopy
from typing import Any, Callable, List, Optional, Sequence, Type, Union, Dict, Tuple

import gym
import numpy as np

from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvStepReturn


class SPGD(object):
    r"""Implements stochastic parallel gradient descent (optionally with momentum) for coherent pulse stacking
    """

    def __init__(self, lr, momentum=0.,dampening=0,nesterov=False):
        if  lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        self.defaults= dict(lr=lr, momentum=momentum,dampening=dampening,nesterov=nesterov)
        self.state = {'step':0}

    def step(self, grad):
        """Performs a single optimization step.
        """
        lr = self.defaults['lr']
        momentum = self.defaults['momentum']
        dampening = self.defaults['dampening']
        nesterov = self.defaults['nesterov']

        d_p = grad
        if momentum != 0:
            if 'momentum_buffer' not in self.state:
                buf = d_p
            else:
                buf = self.state['momentum_buffer']
                buf = buf*momentum + (1-dampening)*d_p
            self.state['momentum_buffer'] = buf
            if nesterov:
                d_p = d_p + momentum*buf
            else:
                d_p = buf

        delta = lr * d_p
        self.state['step'] +=1

        return delta

class SPGDVecEnv(DummyVecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], spgd_factor: float = 0., spgd_begin: int = 1,
                 spgd_warmup: int = 1, spgd_lr: float = 100.,spgd_momentum: float = 0.):
        super(SPGDVecEnv, self).__init__(env_fns)
        self.spgd_factor = spgd_factor
        self.spgd_begin =spgd_begin
        self.spgd_warmup = spgd_warmup

        if self.spgd_factor>0:
            self.spgd_optim = SPGD(lr=spgd_lr, momentum=spgd_momentum)
        else:
            self.spgd_optim = None

        self.spgd_step_number = 0


    def pick_spgd_action(self, last_info : Dict = None) -> Tuple:
        if self.spgd_factor<=0 or self.spgd_step_number < self.spgd_begin or 'grad' not in last_info:
            if self.spgd_factor>0 and 'grad' in last_info:
                spgd_act = self.spgd_optim.step(last_info['grad'])
            return 0,0
        lbd = self.spgd_factor
        if self.spgd_step_number<self.spgd_warmup:
            lbd = lbd * (self.spgd_step_number-self.spgd_begin)/(self.spgd_warmup - self.spgd_begin)
        spgd_act = self.spgd_optim.step(last_info['grad'])
        return lbd, spgd_act

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            act = self.actions[env_idx]
            if self.spgd_factor>0:
                lbd, spgd_act = self.pick_spgd_action(self.buf_infos[env_idx])
                act = (1 - lbd) * act + lbd * spgd_act
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(act)
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        self.spgd_step_number += 1
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))
