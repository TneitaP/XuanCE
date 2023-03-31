import numpy as np
import torch
import torch.nn as nn
from utils.common import get_config, space2shape
from environment import BasicWrapper,NormActionWrapper,DummyVecEnv,RewardNorm,ObservationNorm,DMControl
from representation import MLP
from policy import Gaussian_ActorCritic


def get_action(policy:Gaussian_ActorCritic, inputs):
    '''

    '''
    # agent.interact (PPO)
    #   policy.forward (Gaussian_ActorCritic)
    #       policy.representation.model.forward (MLP)
    tensor_observation = torch.as_tensor(inputs['observation'],dtype=torch.float32,device=policy.actor.device)
    tensor_state = policy.representation.model.forward(tensor_observation)
    a_param,a_dist = policy.actor.forward(tensor_state)
    tensor_action = a_dist.sample().detach().cpu().numpy()

    return tensor_action

if __name__ == '__main__':

    # define hyper-parameters
    nenvs = 1 # Parallel
    device = "cpu"
    test_episode = 1000
    # config = get_config("./config/ppo/", "mujoco")
    envs = [NormActionWrapper(BasicWrapper(DMControl("walker","stand",200))) for i in range(nenvs)]
    envs = DummyVecEnv(envs)
    # envs = RewardNorm(config,envs,train=False)
    # envs = ObservationNorm(config,envs,train=False)

    representation = MLP(space2shape(envs.observation_space),(128,128),nn.LeakyReLU,nn.init.orthogonal,device)
    policy = Gaussian_ActorCritic(envs.action_space,representation,nn.init.orthogonal,device)
    # from pre-trained 
    # policy.load_state_dict(torch.load(r"D:\zzm_codes\XuanCE\_old0329\models\Walker(1)\ppo\model-Wed Mar 29 21_18_55 2023-18000.pth")) # fisrt good
    policy.load_state_dict(torch.load(r"D:\zzm_codes\XuanCE\models\Walker(1)Stand\ppo\model-Thu Mar 30 22_07_48 2023-2000.pth"))
    obs_npy_path = r"D:\zzm_codes\XuanCE\_old0329\models\Walker(1)\ppo\observation_stat.npy"
    obs_rms_data = np.load(obs_npy_path,allow_pickle=True).item()
    scale_factor = np.clip(1/(np.sqrt(obs_rms_data['var']['observation']) + 1e-7),0.1,10)
    
    obs,infos = envs.reset() # (nenvs, 24)
    current_episode = 0
    while current_episode < test_episode:
        print("[%03d]"%(current_episode))
        envs.render("human")
        obs['observation'] = np.clip((obs['observation'] - obs_rms_data['mean']['observation']) * scale_factor,-5, 5) # ObservationNorm performed by ObservationNorm(config,envs,train=False)

        actions = get_action(policy, obs) # (nenvs, 6)
        next_obs,rewards,terminals,trunctions,infos = envs.step(actions)
        for i in range(nenvs):
            if terminals[i] == True or trunctions[i] == True: current_episode += 1
        obs = next_obs