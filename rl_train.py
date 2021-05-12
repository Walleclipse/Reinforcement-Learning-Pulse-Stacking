import os
import json
import argparse
import numpy as np

from agents.actor_critic_agents.SAC import SAC
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.TD3 import TD3
from agents.Trainer import Trainer
from utilities import init_config,init_trial_path
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = "5"
print('pid:',os.getpid())

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='cps',choices=['MountainCarContinuous-v0','cps'])
    # cps-env
    parser.add_argument('--stage', type=int, default=3)
    parser.add_argument('--noise-sigma', type=float, default=5e-4)
    parser.add_argument('--obs-noise-sigma', type=float, default=1e-4)
    parser.add_argument('--init-nonoptimal', type=float, default=None) # 0.1
    parser.add_argument('--action-scale', type=float, default=0.01)
    parser.add_argument('--obs-step', type=int, default=1)
    parser.add_argument('--obs-signal', type=str, nargs='+', default=['power','pzm', 'pulse'],
                        choices=['power', 'pulse', 'pzm'])
    parser.add_argument('--obs-signal-mark', type=int,  default=None)
    parser.add_argument('--normalized', type=int, default=1)
    parser.add_argument('--max-pzm', type=float, default=1)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--max-steps', type=float, default=200)
    parser.add_argument('--reward-threshold', type=float, default=0)
    parser.add_argument('--spgd', type=int, default=1)
    parser.add_argument('--perturb-scale', type=float, default=1e-3)
    # model
    parser.add_argument('--agent', type=str, default='sac',choices=['td3','ddpg','sac'])
    parser.add_argument('--buffer-size', type=int, default=20000) #20000
    parser.add_argument('--actor-lr', type=float, default=5e-4) # 1e-4 ~ 5e-4
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    parser.add_argument('--discount-rate', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--action-noise', type=float, default=0.2)
    parser.add_argument('--noise-clip', type=float, default=0.5) # 0.5
    parser.add_argument('--update-actor-freq', type=int, default=10) # 10
    parser.add_argument('--theta', type=float, default=0.2) # 0.15
    parser.add_argument('--sigma', type=float, default=0.2) # 0.25
    # spgd
    parser.add_argument('--spgd-factor', type=float, default=0.1)  # 0.25
    parser.add_argument('--spgd-begin', type=int, default=2)
    parser.add_argument('--spgd-warmup', type=int, default=10)
    parser.add_argument('--spgd-lr', type=float, default=100)
    parser.add_argument('--spgd-momentum', type=float, default=0.)
    # training
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--update-per-session', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--hidden-dim', type=int, default=400) # 200~400
    parser.add_argument('--hidden-layer', type=int, default=3) # 2~3
    parser.add_argument('--runs-per-agent', type=int, default=1)
    parser.add_argument('--logdir', type=str, default='results') #'/ssd3/abu/code/pulse_stacking/rl_cps/results'
    parser.add_argument('--visualize', type=float, default=1)
    parser.add_argument('--test_render', type=float, default=1)
    parser.add_argument('--save_per_episode', type=int, default=100)

    args = parser.parse_known_args()[0]
    orig_init = args.init_nonoptimal
    if args.stage>=9:
        args.hidden_dim=500
        args.hidden_layer=3
        args.episodes = 600
        args.init_nonoptimal=0.1 
    elif args.stage==8:
        args.hidden_dim=400
        args.hidden_layer=3
        args.episodes = 500
        args.init_nonoptimal=0.11   
    elif args.stage==7:
        args.hidden_dim=400
        args.hidden_layer=3 # 3
        args.episodes = 500 # 500
        args.init_nonoptimal=0.12
    elif args.stage==6:
        args.hidden_dim=300
        args.hidden_layer=3 # 3
        args.episodes = 400
        args.init_nonoptimal=0.13
    elif args.stage>=4:
        args.hidden_dim=300
        args.hidden_layer=2  
        args.episodes = 300
        args.init_nonoptimal=0.16
    elif args.stage==3:
        args.hidden_dim=200
        args.hidden_layer=2  
        args.episodes = 150
        args.init_nonoptimal=0.18
    else:
        args.hidden_dim=100
        args.hidden_layer=2  
        args.episodes = 100
        args.init_nonoptimal=0.2
    if orig_init is not None and orig_init>0:
        args.init_nonoptimal = orig_init
    if args.obs_signal_mark==0:
        args.obs_signal= ['power','pzm', 'pulse']
    elif args.obs_signal_mark==1:
        args.obs_signal= ['pzm', 'pulse']
    elif args.obs_signal_mark==2:
        args.obs_signal= ['power','pzm']
    elif args.obs_signal_mark==3:
        args.obs_signal= ['power']
    elif args.obs_signal_mark==4:
        args.obs_signal= ['pzm']
    elif args.obs_signal_mark==5:
        args.obs_signal= ['pulse']
    args.hidden_sizes = [int(args.hidden_dim)] * int(args.hidden_layer)
    return args, parser

def get_agent(agent):
    agent=agent.lower()
    if agent=='td3':
        return TD3
    if agent=='ddpg':
        return DDPG
    if agent=='sac':
        return SAC
    return None

def run_train(args):
    save_path = os.path.join(args.logdir, args.agent+'@cps'+str(args.stage))
    args.save_path = init_trial_path(save_path)
    with open(os.path.join(args.save_path, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f)
    print(args)
    config=init_config(args)
    agent = get_agent(args.agent)
    trainer = Trainer(config, [agent])
    results = trainer.run_games_for_agents()
    agent_name = agent.agent_name
    res_age=np.array(results[agent_name][0][0])
    score = (max(res_age)+np.mean(res_age[-10:]))/2
    converge_step_1=np.where(res_age>0)[0]
    converge_step_2= np.where(res_age > 100)[0]
    if len(converge_step_1)>0:
        converge_step_1 =converge_step_1[0]
    else:
        converge_step_1 = -1
    if len(converge_step_2)>0:
        converge_step_2 =converge_step_2[0]
    else:
        converge_step_2 = -1
    print("Train Done")
    print(f"Agent={agent_name}, Score={score}, Path={args.save_path}")
    print(f"Final Reward={np.mean(res_age[-10:])}, Converge Step 0 ={converge_step_1}, Converge Step 100 ={converge_step_2},")
    agent = trainer.trained_agents[0][0]
    return agent

def main():
    args,_ = get_args()
    agent = run_train(args)
    eval_results = agent.evaluate(reward_threshold=0.995,max_steps=200)
    rewards=np.array(eval_results['reward'])
    final_reward=np.mean(rewards[-10:])
    converge_step_1=np.where(rewards>0.9)[0]
    converge_step_2= np.where(rewards > 0.97)[0]
    if len(converge_step_1)>0:
        converge_step_1 =converge_step_1[0]
    else:
        converge_step_1 = -1
    if len(converge_step_2)>0:
        converge_step_2 =converge_step_2[0]
    else:
        converge_step_2 = -1
    print("Evaluate Done")
    print(f"Final Reward={final_reward}, Converge Step 90% ={converge_step_1}, Converge Step 95% ={converge_step_2},")

if __name__ == '__main__':
    main()
