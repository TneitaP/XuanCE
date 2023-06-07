import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from enum import Enum

import gym
from utils.common import space2shape,get_config
from environment import BasicWrapper,NormActionWrapper,DummyVecEnv,RewardNorm,ObservationNorm,DMControl
from representation import MLP
from policy import Categorical_ActorCritic,Gaussian_ActorCritic
from learner import PPO_Learner
from agent import PPO_Agent

class ConvertMode(Enum):
    PT = 0
    ONNX = 1

to_cpu = lambda tensor: tensor.detach().cpu().numpy()


class ActorCritic_inference(nn.Module):
    def __init__(self, policy:Gaussian_ActorCritic, obs_rms_data, scale_range=(0.1,10),obs_range=(-5,5)) -> None:
        super(ActorCritic_inference, self).__init__()

        self.rep_model = policy.representation.model
        self.act_model = policy.actor

        self.obs_scale = torch.from_numpy(np.clip(1/(np.sqrt(obs_rms_data['var']['observation']) + 1e-7),scale_range[0],scale_range[1])).float().to(policy.actor.device)
        self.obs_mean = torch.from_numpy(obs_rms_data['mean']['observation']).float().to(policy.actor.device)
        self.obs_range = obs_range
    
    def forward(self, tensor_observation):
        
        tensor_obsnorm = torch.clip((tensor_observation - self.obs_mean) * self.obs_scale, self.obs_range[0], self.obs_range[1])
        tensor_state = self.rep_model.forward(tensor_obsnorm)
        a_param,a_dist = self.act_model.forward(tensor_state)
        tensor_action = a_dist.sample()
        return tensor_action


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain_name",type=str, default="walker")
    parser.add_argument("--task_name",type=str, default="stand")
    parser.add_argument("--time_limit",type=int, default=200)

    parser.add_argument("--config",type=str, default="config\ppo\mujoco_walkerStand.yaml") # training details

    parser.add_argument("--out_type",type=str, default=".onnx")
    parser.add_argument("--pretrain_weight",type=str)
    parser.add_argument("--pretrain_observe_norm",type=str)
    parser.add_argument("--evaluate",type=bool,default=True)
    
    args = parser.parse_known_args()[0]
    return args



####################################################  convert scripts #################################################### 
def get_convert_mode_from_filename(filename):

    if ".onnx" in filename: return ConvertMode.ONNX
    elif ".pt" in filename: return ConvertMode.PT

    print("unsupported model type")
    return 

def convert_ppo_actor(pretrain_weight_filename:str, pretrain_observenorm_filename:str, convert_actnet_filename, flag_evaluate:bool):

    convert_mode = get_convert_mode_from_filename(convert_actnet_filename)
    device = "cuda:0"
    test_episode = 1000
    # config = get_config("./config/ppo/", "mujoco")
    env = NormActionWrapper(BasicWrapper(DMControl("walker","stand",200)))
    
    representation = MLP(space2shape(env.observation_space),(128,128),nn.LeakyReLU,nn.init.orthogonal,device)
    policy = Gaussian_ActorCritic(env.action_space,representation,nn.init.orthogonal,device)
    policy.load_state_dict(torch.load(pretrain_weight_filename))
    policy.eval()

    obs_rms_data = np.load(pretrain_observenorm_filename,allow_pickle=True).item()

    poli_inf = ActorCritic_inference(policy, obs_rms_data)
    poli_inf.eval()

    batch_size = 1
    obs_Tsor = torch.randn(batch_size, space2shape(env.observation_space)['observation'][0]).to(policy.actor.device)
    act_Tsor = poli_inf.forward(obs_Tsor)

    print("test observe shape: ", obs_Tsor.shape)
    print("test action shape: ", act_Tsor.shape)

    print("execute converting ...", end=" ")

    # "walker-stand-ppo-Thu Mar 30 22_03_44 2023-1000_0406_3.onnx"
    if convert_mode == ConvertMode.ONNX:
        torch.onnx.export(model = poli_inf, args = obs_Tsor, f = convert_actnet_filename, 
                # export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch'},    # variable length axes
                                'output' : {0 : 'batch'}})

    elif convert_mode == ConvertMode.PT:
        traced_script_module = torch.jit.trace(poli_inf, obs_Tsor) 
        traced_script_module.save(convert_actnet_filename)

    print("succeed!")
    
    if flag_evaluate: evaluate_ppo_actor(convert_mode, convert_actnet_filename)

def evaluate_ppo_actor(convert_mode:ConvertMode, saved_filename):

    if convert_mode == ConvertMode.ONNX:
        import onnx 
        import onnxruntime
        onnx_model = onnx.load(saved_filename)
        onnx.checker.check_model(onnx_model)

    elif convert_mode == ConvertMode.PT:
        poli_inf_pt = torch.jit.load(saved_filename)
        poli_inf_pt.eval()
        print(poli_inf_pt)

    
    nenvs = 4 # Parallel
    device = "cuda:0" #"cpu"
    test_episode = 1000
    # config = get_config("./config/ppo/", "mujoco")
    envs = [BasicWrapper(DMControl("walker","stand",200)) for i in range(nenvs)]
    # envs = [BasicWrapper(DMControl("walker","stand",200)) for i in range(nenvs)]
    envs = DummyVecEnv(envs)

    poli_inf_ort = onnxruntime.InferenceSession(saved_filename)

    obs,infos = envs.reset() # (nenvs, 24)
    current_episode = 0
    while current_episode < test_episode:
        print("[%03d]"%(current_episode))
        envs.render("human")
        
        act_Arr = None
        if convert_mode == ConvertMode.ONNX:
            ort_inputs = {poli_inf_ort.get_inputs()[0].name: obs['observation'].astype(np.float32)}
            act_Arr = poli_inf_ort.run(None, ort_inputs)[0]

        elif convert_mode == ConvertMode.PT:
            ptin_Tsor = torch.from_numpy(obs['observation'].astype(np.float32)).cuda()
            act_Tsor = poli_inf_pt.forward(ptin_Tsor)
            act_Arr = to_cpu(act_Tsor)

        next_obs,rewards,terminals,trunctions,infos = envs.step(act_Arr)
        for i in range(nenvs):
            if terminals[i] == True or trunctions[i] == True: current_episode += 1
        obs = next_obs



if __name__ == "__main__":
    
    # convert_ppo_actor(pretrain_weight_filename = r"D:\zzm_codes\XuanCE\models\Walker(2)Walk\ppo\walker-walk-ppo-05-08-13_22_19-2023-23000.pth", 
    #                   pretrain_observenorm_filename = r"D:\zzm_codes\XuanCE\models\Walker(2)Walk\ppo\observation_stat.npy", 
    #                   convert_actnet_filename = r"D:\zzm_codes\XuanCE\models\Walker(2)Walk\ppo\walker-walk-ppo-05-08-13_22_19-2023-23000.onnx", 
    #                   flag_evaluate = True
    #                   )

    args = get_args()

    convert_ppo_actor(pretrain_weight_filename = args.pretrain_weight, 
                      pretrain_observenorm_filename = args.pretrain_observe_norm, 
                      convert_actnet_filename = (args.pretrain_weight).replace('.pth', args.out_type), 
                      flag_evaluate = args.evaluate
                      )