import os
import argparse
import time

import numpy as np
import pandas as pd
from environments import CPS_env
from utilities import process_hyperparams, split_hyperparams, StoreDict
from utilities import create_callback, create_logger, create_envs
from utilities import create_model, create_action_noise,load_pretrained_agent, save_agent
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser()
    root_dir='./'
    # cps-env
    parser.add_argument('--stage', type=int, default=5) #
    parser.add_argument('--difficulty', type=str, default='hard', choices=['easy','medium','hard','custom']) # 0.1
    parser.add_argument('--max_episode_steps', type=float, default=200)
    parser.add_argument('--frame_stack', type=int, default=None)
    parser.add_argument("--vec-env", help="VecEnv type", type=str, default="spgd", choices=["dummy", "subproc", 'spgd'])
    # model
    parser.add_argument('--algo', type=str, default='td3',choices=['td3','ddpg','sac','ppo','a2c'])
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    # training
    parser.add_argument('--n_timesteps', type=int, default=50000) # 50000
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--deterministic_eval', type=bool,  default=True)
    # misc
    parser.add_argument('--default_hp_path', type=str, default=root_dir+'utilities/default_config.yml')
    parser.add_argument('--out_dir', type=str, default=root_dir+'results')
    parser.add_argument("--eval-freq", help="Evaluate the agent every n steps (if negative, no evaluation)", default=2000, type=int)
    parser.add_argument("--n-eval-episodes", help="Number of episodes to use for evaluation", default=3, type=int)
    parser.add_argument("--save-freq", help="Save the model every n steps (if negative, no checkpoint)", default=-1,
                        type=int)
    parser.add_argument("-tb", "--tensorboard-log", help="Tensorboard log dir", default="results/", type=str)
    parser.add_argument("-i", "--trained-agent", help="Path to a pretrained agent to continue training", default="",
                        type=str)
    parser.add_argument("--verbose", help="Verbose mode (0: no output, 1: INFO)", default=1, type=int)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument("--hyperparams",#default={'policy_kwargs':"dict(net_arch=[256, 256,256])",'learning_rate':0.001},
        type=str,nargs="+",action=StoreDict, help="Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)",)
    parser.add_argument('--device', type=str, default='cuda:1')

    args = parser.parse_args()
    return args


def train(args):
    args.env_id = CPS_env
    hyperparams = process_hyperparams(args)
    if args.algo=='td3' and 'episodic' in hyperparams:
        if hyperparams['episodic']:
            hyperparams['train_freq'] = (1, "episode")
            hyperparams['gradient_steps'] =-1
        del hyperparams['episodic']
    hyperparams = create_action_noise(hyperparams)
    algo_hps, env_hps, exp_hps, spgd_hps = split_hyperparams(hyperparams)

    callback = create_callback(exp_hps, env_hps)
    algo_hps['default_logger']=create_logger(exp_hps)
    env = create_envs(exp_hps,env_hps, eval_env=False)

    if exp_hps['trained_agent'] !='' and os.path.isfile(exp_hps['trained_agent']):
        model = load_pretrained_agent(exp_hps,env,algo_hps)
    else:
        model = create_model(exp_hps['algo'],env,algo_hps)

    log_interval=min(100,max(1,hyperparams['eval_freq']//hyperparams['max_episode_steps']))
    model.learn(exp_hps['n_timesteps'],callback=callback,log_interval=log_interval)

    model_path = save_agent(model, exp_hps)
    del model, env, callback

    model = load_pretrained_agent(exp_hps,model_path=model_path)
    env = create_envs(exp_hps,env_hps, eval_env=False)
    st = time.time()
    obs = env.reset()
    print('env reset time:', time.time()-st)
    reward_list=[]
    pred_times, env_times=[],[]
    for ii in range(200):
        st=time.time()
        action, _states = model.predict(obs, deterministic=exp_hps['deterministic_eval'])
        pred_times.append(time.time()-st)
        st = time.time()
        obs, rewards, dones, info = env.step(action)
        env_times.append(time.time() - st)
        #env.render()
        reward_list.append(rewards)
        if dones:
            break

    reward_list=np.array(reward_list)
    print(f'Max Reward:{reward_list.max()}, Avg Reward:{reward_list.mean()}')
    print('total pred step:', len(pred_times), ', average pred_times:', np.mean(pred_times))
    print('total environment step:', len(env_times), ', average environment_times:', np.mean(env_times))


    joblib.dump(reward_list, exp_hps['log_path']+ '/test_reward_list_temp.pkl')

    env.close()
    plt.figure()
    y = reward_list
    plt.plot(np.arange(len(y)), y)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('Evaluation Rewards')
    plt.savefig(exp_hps['log_path']+'/final_test.png')
    #plt.show()
    plt.close()

    df = pd.read_csv(exp_hps['log_path']+'/results.csv')
    plt.figure()
    y = df['rollout/ep_rew_mean'].values
    x = np.arange(len(y))
    plt.plot(x, y)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('Training Rewards')
    plt.savefig(exp_hps['log_path'] + '/train_reward.png')
    #plt.show()
    plt.close()


def main():
    args = get_args()
    train(args)

if __name__=='__main__':
    main()


