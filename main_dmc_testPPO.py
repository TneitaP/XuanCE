import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
from utils.common import space2shape,get_config
from environment import BasicWrapper,NormActionWrapper,DummyVecEnv,RewardNorm,ObservationNorm,DMControl
from representation import MLP
from policy import Categorical_ActorCritic,Gaussian_ActorCritic
from learner import PPO_Learner
from agent import PPO_Agent

to_cpu = lambda tensor: tensor.detach().cpu().numpy()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name",type=str)
    parser.add_argument("--task_name",type=str)
    parser.add_argument("--time_limit",type=int)

    parser.add_argument("--config",type=str)
    parser.add_argument("--device",type=str,default="cuda:0")
    
    parser.add_argument("--pretrain_weight",type=str)
    parser.add_argument("--pretrain_reward_norm",type=str)
    parser.add_argument("--pretrain_observe_norm",type=str)
    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    args = get_args()
    device = args.device
    config = get_config(args.config)
    config.nenvs = 2 # over-write
    envs = [NormActionWrapper(BasicWrapper(DMControl(args.domain_name,args.task_name, args.time_limit))) for i in range(config.nenvs)]

    envs = DummyVecEnv(envs)
    envs = RewardNorm(config,envs,train=True,pretrain_para_path=args.pretrain_reward_norm)
    envs = ObservationNorm(config,envs,train=True,pretrain_para_path=args.pretrain_observe_norm)

    mlp_hiddens = tuple(map(int, config.mlp_hiddens.split(",")))
    representation = MLP(space2shape(envs.observation_space),mlp_hiddens,nn.LeakyReLU,nn.init.orthogonal_,device)
    policy = Gaussian_ActorCritic(envs.action_space,representation,nn.init.orthogonal_,device)

    policy.load_state_dict(torch.load(args.pretrain_weight,map_location=device))
    policy.eval()

    obs,infos = envs.reset() # (nenvs, 24)

    test_episode = 1000
    current_episode = 0
    while current_episode < test_episode:
        print("[%03d]"%(current_episode))
        envs.render("human")
        # obs_Tsor = torch.from_numpy(obs['observation']).float().to(policy.actor.device)
        _,act_Distrib,_ = policy.forward(obs) # (nenvs, 6)
        act_Tsor = act_Distrib.sample()
        next_obs,rewards,terminals,trunctions,infos = envs.step(to_cpu(act_Tsor))
        for i in range(config.nenvs):
            if terminals[i] == True or trunctions[i] == True: current_episode += 1
        obs = next_obs