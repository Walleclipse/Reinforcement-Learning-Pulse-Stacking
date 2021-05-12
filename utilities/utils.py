import os
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
import gym
import torch

from .data_structures.Config import Config
from environments import CPS_env
from environments.Open_AI_Wrappers import TimeLimit

def init_config(args):
	config = Config()
	config.seed = args.seed
	if args.task == 'cps':
		env = CPS_env(stage=args.stage, noise_sigma=args.noise_sigma, obs_noise_sigma=args.obs_noise_sigma,
		              init_nonoptimal=args.init_nonoptimal,action_scale=args.action_scale,obs_step=args.obs_step,
		              obs_signal=args.obs_signal,normalized_action=args.normalized,normalized_observation=args.normalized,
		              max_pzm=args.max_pzm,seed=args.seed,reward_threshold=args.reward_threshold,
		              spgd=args.spgd, perturb_scale=args.perturb_scale)
		env = TimeLimit(env, max_episode_steps=args.max_steps)
	else:
		env = gym.make(args.task)
	config.environment = env

	config.num_episodes_to_run = args.episodes
	config.file_to_save_data_results = os.path.join(args.save_path, 'data_results.pkl')
	config.file_to_save_results_graph = os.path.join(args.save_path, 'results_graph.png')

	config.show_solution_score = False # args.render>0
	config.visualise_individual_results = False# args.render>0
	config.visualise_overall_agent_results = args.visualize
	config.test_render = args.test_render
	config.save_per_episode = args.save_per_episode
	config.standard_deviation_results = 1.0
	config.runs_per_agent = args.runs_per_agent
	config.use_GPU = torch.cuda.is_available()
	config.overwrite_existing_results_file = False
	config.randomise_random_seed = True
	config.save_model = True

	config.hyperparameters = {
		"Actor_Critic_Agents": {
			"Actor": {
				"learning_rate": args.actor_lr,
				"linear_hidden_units": args.hidden_sizes,
				"final_layer_activation": "TANH",#None,
				"batch_norm": False,
				"tau":  args.tau,
				"gradient_clipping_norm": 5,
				"initialiser": "Xavier",
				#"y_range":(-1,1),  # ()if args.agent=='sac': () else:
			},

			"Critic": {
				"learning_rate": args.critic_lr,
				"linear_hidden_units": args.hidden_sizes,
				"final_layer_activation": None,
				"batch_norm": False,
				"buffer_size": args.buffer_size,
				"tau": args.tau,
				"gradient_clipping_norm": 5,
				"initialiser": "Xavier"
			},

			"min_steps_before_learning": 1000,  # for SAC only
			"batch_size": args.batch_size,
			"discount_rate": args.discount_rate,
			"mu": 0.0,  # for O-H noise
			"theta": args.theta,  # for O-H noise
			"sigma": args.sigma,  # for O-H noise
			"action_noise_std": args.action_noise,  # for TD3
			"action_noise_clipping_range": args.noise_clip,  # for TD3
			"update_every_n_steps":args.update_actor_freq,
			"learning_updates_per_learning_session": args.update_per_session,
			"automatically_tune_entropy_hyperparameter": True,
			"entropy_term_weight": None,
			"add_extra_noise": True,
			"do_evaluation_iterations": True,
			"clip_rewards": False,
			"HER_sample_proportion": 0.8,
			"exploration_worker_difference": 1.0,
			"save_path":args.save_path,
			"spgd_factor":args.spgd_factor,
			"spgd_begin": args.spgd_begin,
			"spgd_warmup": args.spgd_warmup,
			"spgd_lr": args.spgd_lr,
			"spgd_momentum": args.spgd_momentum,
		}
	}

	return config

def init_trial_path(logdir):
	"""Initialize the path for a hyperparameter setting
	"""
	os.makedirs(logdir, exist_ok=True)
	trial_id = 0
	path_exists = True
	path_to_results = logdir + '/{:d}'.format(trial_id)
	while path_exists:
		trial_id += 1
		path_to_results = logdir + '/{:d}'.format(trial_id)
		path_exists = os.path.exists(path_to_results)
	save_path = path_to_results
	os.makedirs(save_path, exist_ok=True)
	return save_path

def subplot(R, P, Q, S,save_name=''):
	joblib.dump([R, P, Q, S],save_name.replace('png','pkl'))

	r = list(zip(*R))
	p = list(zip(*P))
	q = list(zip(*Q))
	s = list(zip(*S))
	plt.figure()
	fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

	ax[0, 0].plot(list(r[1]), list(r[0]), 'r')  # row=0, col=0
	ax[1, 0].plot(list(p[1]), list(p[0]), 'b')  # row=1, col=0
	ax[0, 1].plot(list(q[1]), list(q[0]), 'g')  # row=0, col=1
	ax[1, 1].plot(list(s[1]), list(s[0]), 'k')  # row=1, col=1
	ax[0, 0].title.set_text('Reward')
	ax[1, 0].title.set_text('Policy loss')
	ax[0, 1].title.set_text('Q loss')
	ax[1, 1].title.set_text('Max steps')
	plt.savefig(save_name)
	plt.show()

def plot_pulse(pulses=[],save_dir='', name=''):
	plt.figure()

	T_ps = pulses[-1]['T_ps']
	max_power = max(pulses[-1]['I'])
	for ii in range(len(pulses)):
		AT = pulses[ii]['I']/max_power
		plt.plot(T_ps, AT,label='Pulse_trains_'+str(ii))
	plt.legend()

	plt.title('Pulses')
	plt.xlabel('T_ps')
	plt.ylabel('A.U.')
	plt.title(name)
	#plt.ylim((0,1))
	if len(save_dir)>0 and len(name)>0:
		plt.savefig(save_dir+name+'pulse.png')
	plt.show()
	plt.close()