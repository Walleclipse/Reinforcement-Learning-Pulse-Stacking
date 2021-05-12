import math
import random
import io
import numpy as np
import pynlo
from functools import partial
import matplotlib.pyplot as plt
import gym
from gym import spaces
from gym.utils import seeding


from .PulseStacking import init_PZM_l0, StackStage, PhaseModulator, MultiStack, detect, copy_pulse, combine_trains
from .utils import NormalizedAct


class CPS_env(gym.Env):
    metadata = {
        'render.modes': ['human'],
    }
    environment_name = "Delay Line Coherent Pulse Stacking"

    def __init__(self, stage=7, noise_sigma=0.001, obs_noise_sigma=0, init_nonoptimal=0.1, action_scale=0.1,
                 obs_step=1, obs_signal=['power', 'pulse', 'pzm'], normalized_action=False, normalized_observation=True,
                 max_pzm=1,seed=None, max_episode_steps=None, reward_threshold=None,spgd=False,perturb_scale=1e-3,**kwargs):
        '''
        init_state = [ 'optimal','random','non_optimal']
        obs_feat =['power','action','PZM','pulse']
        '''
        self.stage = stage
        self.noise_sigma = noise_sigma  # /stage
        self.obs_noise_sigma = obs_noise_sigma
        self.init_nonoptimal = init_nonoptimal  # /stage
        self.action_scale = action_scale
        self.normalized_action = normalized_action
        self.normalized_observation = normalized_observation
        self.obs_signal = obs_signal
        self.obs_step = max(1, int(obs_step))
        self.max_pzm=max_pzm
        self.reward_threshold = reward_threshold
        self.spgd = spgd
        self.perturb_scale = perturb_scale
        self.max_episode_steps = max_episode_steps
        self.elapsed_steps = 0
        self.viewer = None    
        self.seed(seed)
        self.init_state()

    def _init_pulse(self):
        frep_MHz = 50000  # 5G # 1000
        period = 1e6 / frep_MHz  # =20 ps

        FWHM = 4.  # = 4ps,  pulse duration (ps) # 0.01
        pulseWL = 1030  # pulse central wavelength (nm)
        EPP = 2e-10  # Energy per pulse (J) # 0.1 nj
        GDD = 0.0  # Group delay dispersion (ps^2)
        TOD = 0.0  # Third order dispersion (ps^3)

        # Window = 10.0  # simulation window (ps)
        Window = 2 ** (self.stage) * period #  # simulation window (ps)
        Points = int(2 ** 3 * Window / FWHM)
        # Points = 2 ** 13  # simulation points

        # create the pulse!
        self.pulse = pynlo.light.DerivedPulses.SechPulse(power=1,  # Power will be scaled by set_epp
                                                         T0_ps=FWHM / 1.76,
                                                         center_wavelength_nm=pulseWL,
                                                         time_window_ps=Window,
                                                         GDD=GDD, TOD=TOD,
                                                         NPTS=Points,
                                                         frep_MHz=frep_MHz,
                                                         power_is_avg=False)

        # set the pulse energy!
        self.pulse.set_epp(EPP)

        self.frep_MHz = frep_MHz
        self.pulseWL_mm = pulseWL * 1e-6

    def _init_stackstage(self):
        stacks_list = []

        l0_list = init_PZM_l0(self.stage, self.frep_MHz)

        noise_sigma = self.noise_sigma * self.pulseWL_mm
        init_nois_scale = self.init_nonoptimal * self.pulseWL_mm

        for ii in range(1, self.stage + 1):
            fold = 2
            optim_l0 = l0_list[ii] / fold
            l0 = optim_l0
            if init_nois_scale < 0:
                l0 += ((-1) ** ii) * init_nois_scale
            elif init_nois_scale > 0:
                l0 += self.np_random.normal(loc=0, scale=init_nois_scale, size=1)[0]
            # l0 += np.random.uniform(low=init_nois_scale, high=-init_nois_scale, size=1)[0]

            ss = StackStage(PZM_fold=fold, PZM_l0=l0, optim_l0=optim_l0,
                            noise_sigma=noise_sigma, np_rnd=self.np_random, name='s' + str(ii))
            stacks_list.append(ss)

        self.stacks_list = stacks_list

    def _init_space(self):
        action_scale = self.action_scale * self.pulseWL_mm
        self.space_dict = {}
        # action
        self.space_dict['act/pzm'] = spaces.Box(low=-action_scale, high=action_scale, shape=(self.stage,),
                                                dtype=np.float32)
        self.norm_act_fn = NormalizedAct(action_space=self.space_dict['act/pzm'])
        if self.normalized_action:
            self.action_space = spaces.Box(low=-1, high=1, shape=(self.stage,), dtype=np.float32)
        else:
            self.action_space = self.space_dict['act/pzm']

        # power
        pow_low = [math.floor(self.orig_f2_power)]
        pow_high = [math.ceil(self.max_f2_power)]
        self.space_dict['obs/power'] = spaces.Box(low=np.array(pow_low), high=np.array(pow_high), dtype=np.float32)

        # pulse
        self.peak_per_ind = int(len(self.ideal_pulse.AT) / 2 ** (self.stage))
        dim = len(self.ideal_pulse.AT[self.peak_per_ind::self.peak_per_ind])
        pulse_low = list([0] * dim)
        pulse_high = list([self.max_avg_power] * dim)
        self.space_dict['obs/pulse'] = spaces.Box(low=np.array(pulse_low), high=np.array(pulse_high), dtype=np.float32)

        # pzm
        self.pzm_range = self.max_pzm* self.pulseWL_mm
        pzm_lens = self.pulse_stacking.pzm()
        mean_len = np.sum(pzm_lens) / (2 ** len(pzm_lens) - 1)
        pzm_low = [2 ** i * mean_len - self.pzm_range for i in range(len(pzm_lens))]
        pzm_high = [2 ** i * mean_len + self.pzm_range for i in range(len(pzm_lens))]
        self.space_dict['obs/pzm'] = spaces.Box(low=np.array(pzm_low), high=np.array(pzm_high), dtype=np.float32)

        obs_low, obs_high = [], []
        t_obs_low, t_obs_high = [], []
        for sig in self.obs_signal:
            if sig == 'power':
                obs_low += pow_low
                obs_high += pow_high
                t_obs_low += [math.floor(self.orig_f2_power)*2] # *1
                t_obs_high += [math.ceil(self.max_f2_power) / 2]
            if sig == 'pulse':
                obs_low += pulse_low
                obs_high += pulse_high
                t_obs_low += list([0] * dim)
                t_obs_high += list([self.max_avg_power / self.stage] * dim)
            if sig == 'pzm':
                obs_low += pzm_low
                obs_high += pzm_high
                t_obs_low += [2 ** i * mean_len - self.pzm_range / 2 for i in range(len(pzm_lens))]
                t_obs_high += [2 ** i * mean_len + self.pzm_range / 2 for i in range(len(pzm_lens))]

        obs_space = spaces.Box(low=np.array(t_obs_low), high=np.array(t_obs_high), dtype=np.float32)
        self.norm_obs_fn = NormalizedAct(action_space=obs_space)
        if self.normalized_observation:
            obs_noise_sigma = [1] * len(obs_low)
        else:
            obs_noise_sigma = [math.sqrt(h / 2 - l / 2) for (h, l) in zip(obs_high, obs_low)]

        obs_noise_sigma = self.obs_noise_sigma * np.array(obs_noise_sigma)
        self.obs_noise = partial(self.np_random.normal, loc=0, scale=obs_noise_sigma)

        if self.obs_step > 1:
            obs_low = obs_low * self.obs_step
            obs_high = obs_high * self.obs_step

        if self.normalized_observation:
            low=self.norm_obs_fn.normalize(np.array(obs_low),clip=False)
            high = self.norm_obs_fn.normalize(np.array(obs_high), clip=False)
            self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            self.observation_space = spaces.Box(low=np.array(obs_low), high=np.array(obs_high), dtype=np.float32)

        self.reward_range = (
            math.floor(self._cal_reward(self.orig_f2_power)), math.ceil(self._cal_reward(self.max_f2_power)))

    def init_state(self):
        self._init_pulse()
        self.pm = PhaseModulator(stage=self.stage)
        self._init_stackstage()
        self.pulse_stacking = MultiStack(stacks_list=self.stacks_list)

        self.orig_trains = self.pm.infer(self.pulse)
        orig_pulse_dict = combine_trains(self.orig_trains)
        orig_pulses, orig_f2_power = detect(orig_pulse_dict)
        self.orig_f2_power = round(orig_f2_power, 2)
        self.orig_pulses = orig_pulses

        trains_hist = self.pulse_stacking.infer(self.orig_trains)
        final_pulse = combine_trains(trains_hist[-1])
        initial_pulses, initial_f2_power = detect(final_pulse)
        self.initial_f2_power = round(initial_f2_power, 2)
        self.initial_pulses = initial_pulses

        ideal_pulse = copy_pulse(self.pulse)
        for _ in range(self.stage):
            new_aw = math.sqrt(2) * ideal_pulse.AW
            ideal_pulse.set_AW(new_aw)
        self.ideal_pulse = ideal_pulse
        ret_pulse, f2_power = detect({'p': ideal_pulse, 's': None})
        self.max_f2_power = round(f2_power, 2)
        self.max_avg_power = max(ret_pulse['I'])

        self._init_space()
        self.current_info = None
        self.hist_obs = []

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _prep_observation(self, f2_power, pulse, pzm):
        sample_pulse = pulse['I'][self.peak_per_ind::self.peak_per_ind]
        obs = []
        for sig in self.obs_signal:
            if sig == 'power':
                obs += [f2_power]
            if sig == 'pulse':
                obs += sample_pulse.flatten().tolist()
            if sig == 'pzm':
                obs += pzm
        obs = np.array(obs)
        if self.normalized_observation:
            obs = self.norm_obs_fn.normalize(obs, clip=False)
        if self.obs_noise_sigma > 0:
            obs += self.obs_noise()
        obs = obs.flatten()
        return obs

    def _cal_reward(self, power):
        reward = -(power - self.max_f2_power) ** 2 / (self.max_f2_power * self.initial_f2_power)
        reward = 2*(reward+0.5)
        return reward

    def get_score_to_win(self):
        if self.reward_threshold is not None:
            return self.reward_threshold
        else:
            return self.reward_range[1]

    def extract_observation(self, trains_hist):
        final_pulse = combine_trains(trains_hist[-1])
        ret_pulse, f2_power = detect(final_pulse)
        f2_power = round(f2_power, 2)
        pzm_lens = self.pulse_stacking.pzm()
        reward = self._cal_reward(f2_power)
        info = {'power': f2_power, 'pulse': ret_pulse, 'pzm': pzm_lens, 'reward': reward,
                'delta_pzm': self.pulse_stacking.delta_pzm()}
        done = not self.space_dict['obs/pzm'].contains(pzm_lens)
        self.current_info = info
        observation = self._prep_observation(f2_power, ret_pulse, pzm_lens)
        if self.obs_step > 1:
            self.hist_obs.append(observation)
            self.hist_obs = self.hist_obs[-self.obs_step:]
            observation = np.concatenate(self.hist_obs, axis=0)
        return observation, info, done

    def random_action(self):
        act = self.np_random.uniform(low=self.action_space.low, high=self.action_space.high, size=self.stage)
        return act

    def free_run(self):
        self.pulse_stacking.free_run()

    # self.pulse_stacking.clip_pzm(low=-self.pzm_range, high=self.pzm_range)

    def measure(self, add_noise=True):
        if add_noise:
            self.free_run()
        trains_hist = self.pulse_stacking.infer(self.orig_trains)
        observation, info, done = self.extract_observation(trains_hist)
        return observation, info

    def update(self, action):
        if self.normalized_action:
            action = self.norm_act_fn.reverse_normalize(action, clip=True)
        self.pulse_stacking.feedback(action)


    def perturb(self, perturb_scale=0.001):
        loc = perturb_scale * self.pulseWL_mm
        action = self.np_random.uniform(low=-loc, high=loc, size=self.stage)
        self.pulse_stacking.feedback(action)
        if self.normalized_action:
            action = self.norm_act_fn.normalize(action)
        return action

    def cal_grad(self,prev_observation):
        y_0 = prev_observation[0]

        delta_x = self.perturb(perturb_scale=self.perturb_scale)

        trains_hist = self.pulse_stacking.infer(self.orig_trains)
        final_pulse = combine_trains(trains_hist[-1])
        ret_pulse, f2_power = detect(final_pulse)
        f2_power = round(f2_power, 2)
        pzm_lens = self.pulse_stacking.pzm()
        observation_1 = self._prep_observation(f2_power, ret_pulse, pzm_lens)
        y_1 = observation_1[0]
        grad = (y_1 - y_0) * delta_x

        self.update(-delta_x)

        return grad

    def plot_output(self):
        orig_pulses = self.orig_pulses
        pulses = self.current_info['pulse']
        max_power = self.max_avg_power

        plt.plot(orig_pulses['T_ps'], orig_pulses['I'] / max_power, label='Original_Pulses')
        plt.plot(pulses['T_ps'], pulses['I'] / max_power, label='Stacked_Pulses')
        plt.legend()
        plt.xlabel('T_ps')
        plt.ylabel('A.U.')
        plt.title('Oscilloscope Pulses')
        plt.show()
        print(
            f"Current Power={self.current_info['f2_power']}, Maximum Power={self.max_f2_power}, Original Power={self.orig_f2_power}")
        return

    def print_log(self):
        logs = f"*** Step={self.elapsed_steps}, Reward={self.current_info['reward']} ***\n"
        logs += f"Current Power={self.current_info['power']}, Maximum Power={self.max_f2_power}, Original Power={self.orig_f2_power}"
        print(logs)

    def reset(self):
        self.elapsed_steps = 0
        self._init_stackstage()
        self.pulse_stacking = MultiStack(stacks_list=self.stacks_list)
        trains_hist = self.pulse_stacking.infer(self.orig_trains)
        observation, info, done = self.extract_observation(trains_hist)
        if self.obs_step > 1:
            for _ in range(self.obs_step - 1):
                observation, info = self.measure(add_noise=True)
        return observation

    def step(self, action):
        self.free_run()
        self.update(action)
        trains_hist = self.pulse_stacking.infer(self.orig_trains)
        observation, info, done = self.extract_observation(trains_hist)
        if self.spgd:
            grad = self.cal_grad(observation)
        else:
            grad = None
        info['grad']=grad
        self.elapsed_steps += 1
        if self.max_episode_steps is not None:
            if self.elapsed_steps >= self.max_episode_steps:
                done = True
                info['TimeLimit.truncated'] = True
            else:
                info['TimeLimit.truncated'] = False
        return observation, info['reward'], done, info

    def render(self, mode='human'):
        from .utils import Image
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(600, 400)
            self.viewer.set_bounds(-1.1, 1.1, -1.1, 1.1)
        pulses = self.current_info['pulse']
        s_I = pulses['I'][self.peak_per_ind::self.peak_per_ind]
        s_T = pulses['T_ps'][self.peak_per_ind::self.peak_per_ind]
        plt.figure(figsize=(5, 4))
        plt.plot(pulses['T_ps'], pulses['I'] / self.max_avg_power, color='b')
        plt.scatter(s_T, s_I / self.max_avg_power, marker='*', color='r')
        plt.grid()
        plt.ylim((0, 1))
        plt.xlabel('T_ps')
        plt.ylabel('A.U.')
        plt.title('Oscilloscope Pulses')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image('buf', 2., 2., fileobj=buf)
        self.viewer.add_onetime(img)
        rend_img = self.viewer.render(return_rgb_array=mode == 'rgb_array')
        plt.close()
        return rend_img

    def close(self):
        self.elapsed_steps = 0
        if self.viewer:
            self.viewer.close()
            self.viewer = None
