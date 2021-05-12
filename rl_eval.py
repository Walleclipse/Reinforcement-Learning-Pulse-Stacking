import os
import json
import argparse
import numpy as np
import copy
from utilities import init_config
from rl_train import get_args, get_agent

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "5"
print('pid:', os.getpid())

def get_test_args():
    args, parser = get_args()
    # test-args
    parser.add_argument('--test-model-dir', type=str, default='results/sac@cps5/1')
    parser.add_argument('--test-noise-sigma', type=float, default=None)
    parser.add_argument('--test-init-nonoptimal', type=float, default=0.13)  # 0.1
    parser.add_argument('--test-spgd-factor', type=float, default=0.1)  # 0.25
    parser.add_argument('--test-seed', type=int, default=1)
    parser.add_argument('--test-visualize', type=int, default=1)
    parser.add_argument('--test-spgd-lr', type=float, default=1000)
    parser.add_argument('--test-spgd-momentum', type=float, default=0.1)
    parser.add_argument('--test-rendering', type=int, default=1)

    args = parser.parse_known_args()[0]
    with open(os.path.join(args.test_model_dir, 'args.json'), 'r') as f:
        loaded_args = json.load(f)
    for k, v in loaded_args.items():
        setattr(args, k, v)
    old_dict = args.__dict__.copy()
    for k, v in old_dict.items():
        if k[:5] == 'test_' and v is not None:
            new_k = k[5:]
            setattr(args, new_k, v)
    args.test_render = args.rendering
    args.save_path = args.test_model_dir
    os.makedirs(args.save_path + '/demo_exp/', exist_ok=True)
    args.save_name = 'demo_exp/' + 'test_spgd' + str(args.spgd_factor) + '-' + str(args.spgd_momentum) + '_init' + str(
        args.init_nonoptimal)
    if args.spgd_factor == 1:
        args.spgd_warmup = 2
        args.spgd_begin = 2
    print('loaded args from', os.path.join(args.save_path, 'args.json'))
    print(args)

    return args


def run_eval(args):
    config = init_config(args)
    agent_config = copy.deepcopy(config)
    agent_config.hyperparameters = agent_config.hyperparameters['Actor_Critic_Agents']
    agent_class = get_agent(args.agent)
    agent = agent_class(agent_config)
    agent.load_model(model_path=os.path.join(args.save_path, agent.agent_name + '.pt'))

    eval_results = agent.evaluate(reward_threshold=None, save_name=args.save_name, max_steps=200)

    rewards = np.array(eval_results['reward'])
    final_reward = np.mean(rewards[-10:])
    converge_step_1 = np.where(rewards > 0.9)[0]
    converge_step_2 = np.where(rewards > 0.97)[0]
    if len(converge_step_1) > 0:
        converge_step_1 = converge_step_1[0]
    else:
        converge_step_1 = -1
    if len(converge_step_2) > 0:
        converge_step_2 = converge_step_2[0]
    else:
        converge_step_2 = -1
    print("Evaluate Done")
    print(f"Final Reward={final_reward}, Converge Step 90% ={converge_step_1}, Converge Step 95% ={converge_step_2},")

    return eval_results


def main():
    args = get_test_args()
    eval_results = run_eval(args)


if __name__ == '__main__':
    main()
