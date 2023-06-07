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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name",type=str)
    parser.add_argument("--task_name",type=str)
    parser.add_argument("--time_limit",type=int)

    parser.add_argument("--config",type=str) # training details

    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--pretrain_weight",type=str,default=None)
    parser.add_argument("--pretrain_reward_norm",type=str,default=None)
    parser.add_argument("--pretrain_observe_norm",type=str,default=None)
    parser.add_argument("--render",type=bool,default=False)
    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    args = get_args()
    device = args.device
    config = get_config(args.config)
    envs = [NormActionWrapper(BasicWrapper(DMControl(args.domain_name,args.task_name, args.time_limit))) for i in range(config.nenvs)]
    envs = DummyVecEnv(envs)
    envs = RewardNorm(config,envs,train=True,pretrain_para_path=args.pretrain_reward_norm)
    envs = ObservationNorm(config,envs,train=True,pretrain_para_path=args.pretrain_observe_norm)
    
    mlp_hiddens = tuple(map(int, config.mlp_hiddens.split(",")))
    representation = MLP(space2shape(envs.observation_space),mlp_hiddens,nn.LeakyReLU,nn.init.orthogonal_,device)
    policy = Gaussian_ActorCritic(envs.action_space,representation,nn.init.orthogonal_,device)
    if args.pretrain_weight:
        policy.load_state_dict(torch.load(args.pretrain_weight,map_location=device))
    optimizer = torch.optim.Adam(policy.parameters(),config.lr_rate)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1,total_iters=config.train_steps/config.nsize * config.nepoch * config.nminibatch)
    learner = PPO_Learner(config,policy,optimizer,scheduler,device)
    agent = PPO_Agent(config,envs,policy,learner)
    agent.benchmark(config.train_steps,config.evaluate_steps, render=args.render)




