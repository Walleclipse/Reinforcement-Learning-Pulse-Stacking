import os
import argparse
import yaml
import numpy as np
from environments import CPS_env
from utilities import process_hyperparams, split_hyperparams, StoreDict
from utilities import create_callback, create_logger, create_envs
from utilities import create_model, create_action_noise,load_pretrained_agent, save_agent
import matplotlib.pyplot as plt
import joblib
import  warnings

warnings.filterwarnings("ignore")

def get_args():
    parser = argparse.ArgumentParser()
    root_dir='./'
    # cps-env
    parser.add_argument('--stage', type=int, default=5) #
    parser.add_argument('--difficulty', type=str, default='hard', choices=['easy','medium','hard','custom']) # 0.1
    parser.add_argument('--frame_stack', type=int, default=None)
    parser.add_argument("--vec-env", help="VecEnv type", type=str, default="spgd", choices=["dummy", "subproc", 'spgd'])
    # model
    parser.add_argument('--algo', type=str, default='td3',choices=['td3','ddpg','sac','ppo','a2c'])
    # misc
    parser.add_argument('--default_hp_path', type=str, default=root_dir+'utilities/default_config.yml')
    parser.add_argument('--out_dir', type=str, default=root_dir+'results')
    parser.add_argument('--agent_dir', type=str, default='results/CPS_env_2/td3/3')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()
    return args


def evaluate(args):
    env_diff = args.difficulty
    stage = args.stage
    print('\n Environment: stage=',stage,', difficulty=',env_diff)
    args.env_id = CPS_env
    hyperparams = process_hyperparams(args)
    hyperparams = create_action_noise(hyperparams)
    algo_hps, env_hps, exp_hps, spgd_hps = split_hyperparams(hyperparams,is_save=False)
    trained_dir = args.agent_dir

    trained_hp_path = os.path.join(trained_dir,'hyperparams.yml')
    with open(trained_hp_path, "r") as f:
        loaded_hp = yaml.load(f,Loader=yaml.Loader)
    
    algo_hps.update(loaded_hp['algo'])
    env_hps.update(loaded_hp['env'])
    exp_hps.update(loaded_hp['exp'])
    spgd_hps.update(loaded_hp['spgd'])
    exp_hps['trained_agent'] = os.path.join(trained_dir,'ckpt/trained_agent.zip')

    env_hps['difficulty']=env_diff
    env_hps['stage']=stage
    env = create_envs(exp_hps,env_hps, eval_env=True)

    model = load_pretrained_agent(exp_hps,env,algo_hps)

    obs = env.reset()
    reward_list=[]
    save_fig_dir=os.path.join(exp_hps['log_folder'],'test_results')
    os.makedirs(save_fig_dir, exist_ok=True)
    pulse_list=[]
    for ii in range(200):
        action, _states = model.predict(obs, deterministic=exp_hps['deterministic_eval'])
        obs, rewards, dones, info = env.step(action)
        #env.envs[0].plot_output(os.path.join(save_fig_dir,str(ii)+'.png'))
        reward_list.append(rewards)
        pulse_list.append(info[0]['pulse'])
        if dones:
            break

    reward_list=np.array(reward_list)
    print(f'Max Reward:{reward_list.max()}, Avg Reward:{reward_list.mean()}')

    joblib.dump(pulse_list,save_fig_dir+'/pulse_list.pkl')
    joblib.dump(reward_list, save_fig_dir + f'/spgd_st{stage}_diff{env_diff}_reward_list.pkl')

    env.close()
    plt.figure()
    y = reward_list
    plt.plot(np.arange(len(y)), y)
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid()
    plt.title('Evaluation Rewards')
    plt.savefig(save_fig_dir+'/final_test.png')
    plt.show()


def main():
    args = get_args()
    evaluate(args)

if __name__=='__main__':
    main()


