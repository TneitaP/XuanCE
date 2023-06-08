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

##############################
import os, glob
import numpy as np 
import copy
# from robopianist.suite.tasks import self_actuated_piano, piano_with_shadow_hands
from robopianist import suite
from robopianist import music, viewer
from dm_env_wrappers import CanonicalSpecWrapper
from robopianist.wrappers import PianoSoundVideoWrapper
from mujoco_utils import composer_utils
import dm_env
from dm_control.rl import control

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",type=str,default ="third_party/XuanCE/config/ppo/mujoco_pianoTwinkle.yaml") # training details

    parser.add_argument("--device",type=str,default="cuda:0")
    parser.add_argument("--pretrain_weight",type=str,default=None)
    parser.add_argument("--pretrain_reward_norm",type=str,default=None)
    parser.add_argument("--pretrain_observe_norm",type=str,default=None)
    parser.add_argument("--render",type=bool,default=True)
    args = parser.parse_known_args()[0]
    return args

if __name__ == "__main__":
    args = get_args()
    device = "cuda:0"
    config = get_config(args.config)
    
    
    # suite env:
    # from dm_control import suite
    # env = suite.load(domain_name="walker",task_name="stand")
    
    # task = self_actuated_piano.SelfActuatedPiano(
    #         midi=music.load("TwinkleTwinkleLittleStar"),
    #         # midi=music.load("robopianist/music/data/pig_single_finger/waltz_op_39_no_15-1.proto"),
    #         change_color_on_activation=True,
    #         trim_silence=True,
    #         control_timestep=0.05,
    #     )
    # task = piano_with_shadow_hands.PianoWithShadowHands(
    #     # midi=music.load("robopianist/music/data/self_made/Beethoven-FurElise.mid"),
    #     midi=music.load("TwinkleTwinkleLittleStar"),
    #     change_color_on_activation=True,
    #     trim_silence=True,
    #     control_timestep=0.05,
    #     gravity_compensation=True,
    #     primitive_fingertip_collisions=False,
    #     reduced_action_space=False,
    #     n_steps_lookahead=10,
    #     disable_fingering_reward=False,
    #     disable_forearm_reward=False,
    #     disable_colorization=False,
    #     disable_hand_collisions=False,
    #     attachment_yaw=0.0,
    # )
    

    # env = composer_utils.Environment(
    #         task=task, strip_singleton_obs_buffer_dim=True,
    #         recompile_physics=True
    #     )
    env = suite.load("RoboPianist-repertoire-150-arabesque_no_1-1-v0",
                    midi_file = "robopianist/music/data/pig_single_finger/arabesque_no_1-1.proto")

    '''
    def __init__(self,
               physics,
               task,
               time_limit=float('inf'),
               control_timestep=None,
               n_sub_steps=None,
               flat_observation=False,
               legacy_step: bool = True):
    '''
    
    
    # env = PianoSoundVideoWrapper(env,record_every=1,camera_id="piano/right",record_dir = os.getcwd() + "/tmp/videos0607")


    
    envs = [NormActionWrapper(BasicWrapper(DMControl(env, timelimit=150))) for i in range(config.nenvs)]


    envs = DummyVecEnv(envs)
    envs = RewardNorm(config,envs,train=True,pretrain_para_path=None)
    envs = ObservationNorm(config,envs,train=True,pretrain_para_path=None) 
    # for agent: import copy; copy.deepcopy(envs)

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