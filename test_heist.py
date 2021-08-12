from common.env.procgen_wrappers import *
from common.logger import Logger
from common.storage import Storage
from common.model import NatureModel, ImpalaModel
from common.policy import CategoricalPolicy
from common import set_global_seeds, set_global_log_levels
from agents.ppo import PPO as AGENT
from agents.ppo_aug import PPO_aug as AUG_AGENT

import os, time, yaml, argparse
import gym
from procgen import ProcgenEnv
import random
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='testing', help='experiment name')
    parser.add_argument('--env_name', type=str, default='heist', help='environment ID')
    parser.add_argument('--start_level', type=int, default=int(0), help='start-level for environment')
    parser.add_argument('--num_levels', type=int, default=int(0), help='number of training levels for environment')
    parser.add_argument('--distribution_mode', type=str, default='easy', help='distribution mode for environment')
    parser.add_argument('--param_name', type=str, default='easy', help='hyper-parameter ID')
    parser.add_argument('--device', type=str, default='gpu', required=False, help='whether to use gpu')
    parser.add_argument('--gpu_device', type=int, default=int(0), required=False, help='visible device in CUDA')
    parser.add_argument('--num_timesteps', type=int, default=int(25000000), help='number of training timesteps')
    parser.add_argument('--seed', type=int, default=random.randint(0, 9999), help='Random generator seed')
    parser.add_argument('--log_level', type=int, default=int(40), help='[10,20,30,40]')
    parser.add_argument('--num_checkpoints', type=int, default=int(1), help='number of checkpoints to store')

    args = parser.parse_args()
    exp_name = args.exp_name
    env_name = args.env_name
    start_level = args.start_level
    num_levels = args.num_levels
    distribution_mode = args.distribution_mode
    param_name = args.param_name
    device = args.device
    gpu_device = args.gpu_device
    num_timesteps = args.num_timesteps
    seed = args.seed
    log_level = args.log_level
    num_checkpoints = args.num_checkpoints

    set_global_seeds(seed)
    set_global_log_levels(log_level)

    ####################
    ## HYPERPARAMETERS #
    ####################
    print('[LOADING HYPERPARAMETERS...]')
    with open('hyperparams/procgen/config.yml', 'r') as f:
        hyperparameters = yaml.safe_load(f)[param_name]
    for key, value in hyperparameters.items():
        print(key, ':', value)

    ############
    ## DEVICE ##
    ############
    # gpu
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_device)  # str(gpu_device)
    device = torch.device('cuda')

    # cpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # device = torch.device('cpu')

    #################
    ## ENVIRONMENT ##
    #################
    print('INITIALIZAING ENVIRONMENTS...')
    n_steps = hyperparameters.get('n_steps', 256)
    n_envs = hyperparameters.get('n_envs', 64)
    # By default, pytorch utilizes multi-threaded cpu
    # Procgen is able to handle thousand of steps on a single core
    # torch.set_num_threads(1) # todo: num of threads 1?
    env = ProcgenEnv(num_envs=n_envs,
                     env_name=env_name,
                     start_level=start_level,
                     num_levels=num_levels,
                     distribution_mode=distribution_mode)
    normalize_rew = hyperparameters.get('normalize_rew', True)
    env = VecExtractDictObs(env, "rgb")
    if normalize_rew:
        env = VecNormalize(env, ob=False)  # normalizing returns, but not the img frames.
    env = TransposeFrame(env)
    env = ScaledFloatFrame(env)

    ###########
    ## MODEL ##
    ###########
    print('INTIALIZING MODEL...')
    observation_space = env.observation_space
    observation_shape = observation_space.shape
    architecture = hyperparameters.get('architecture', 'impala')
    in_channels = observation_shape[0]
    action_space = env.action_space

    # Model architecture
    if architecture == 'nature':
        model = NatureModel(in_channels=in_channels)
    elif architecture == 'impala':
        model = ImpalaModel(in_channels=in_channels)

    # Discrete action space
    recurrent = hyperparameters.get('recurrent', False)
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        policy = CategoricalPolicy(model, recurrent, action_size)
    else:
        raise NotImplementedError
    policy.to(device)

    #############
    ## STORAGE ##
    #############
    print('INITIALIZAING STORAGE...')
    hidden_state_dim = model.output_dim
    storage = Storage(observation_shape, hidden_state_dim, n_steps, n_envs, device)

    # REST (added)
    log_test_path = "logs/testing/" + env_name + "/"
    log_path = "logs/training/" + env_name + "/"

    # for loop over algo's (4) and for loop over models (8)
    # for i, dir in enumerate(["baseline", "data_aug", "disout", "disout_data_aug"]):
    for i, dir in enumerate(["disout", "disout_data_aug"]):
        DIR = os.path.join(log_path, dir)
        # loop over files in dir
        for PATH in os.listdir(DIR):
            # PATH (seed dir)
            PATH = os.path.join(DIR, PATH)

            log_save_dir = os.path.join(log_test_path,PATH)

            ############
            ## LOGGER ##
            ############
            # print('INITIALIZAING LOGGER...')
            logger = Logger(n_envs, log_save_dir, True)

            for file in os.listdir(PATH):
                file = os.path.join(PATH, file)
                if file.endswith(".pth"):
                    policy.load_state_dict(torch.load(file)["state_dict"])
                    # policy.load_state_dict(torch.load(file, map_location=torch.device('cpu'))["state_dict"])
                    # policy.eval()

                    ###########
                    ## AGENT ##
                    ###########
                    # print('INTIALIZING AGENT...')
                    if i >= 2:
                        agent = AUG_AGENT(env, policy, logger, storage, device, num_checkpoints, **hyperparameters)
                    else:
                        agent = AGENT(env, policy, logger, storage, device, num_checkpoints, **hyperparameters)

                    #############
                    ## TESTING ##
                    #############
                    # print('START TESTING...')
                    agent.test(num_timesteps)
