import numpy as np
import torch
import torch.nn as nn
import torch.onnx

from utils.common import space2shape,get_config
from environment import BasicWrapper,NormActionWrapper,DummyVecEnv,RewardNorm,ObservationNorm,DMControl
from representation import MLP
from policy import Categorical_ActorCritic,Gaussian_ActorCritic


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


def convert_onnx():
    nenvs = 1 # Parallel
    device = "cpu"
    test_episode = 1000
    # config = get_config("./config/ppo/", "mujoco")
    envs = [NormActionWrapper(BasicWrapper(DMControl("walker","stand",200))) for i in range(nenvs)]
    envs = DummyVecEnv(envs)


    representation = MLP(space2shape(envs.observation_space),(128,128),nn.LeakyReLU,nn.init.orthogonal,device)
    policy = Gaussian_ActorCritic(envs.action_space,representation,nn.init.orthogonal,device)
    policy.eval()

    # policy.load_state_dict(torch.load(r"D:\zzm_codes\XuanCE\models\Walker(1)Stand\ppo\model-Thu Mar 30 22_07_48 2023-2000.pth"))
    policy.load_state_dict(torch.load(r"D:\zzm_codes\XuanCE\models\Walker(1)Stand\ppo\model-Thu Mar 30 22_03_44 2023-1000.pth"))

    obs_npy_path = r"D:\zzm_codes\XuanCE\_old0329\models\Walker(1)\ppo\observation_stat.npy"
    obs_rms_data = np.load(obs_npy_path,allow_pickle=True).item()

    poli_inf = ActorCritic_inference(policy, obs_rms_data)
    poli_inf.eval()
    

    batch_size = 5
    obs_Tsor = torch.randn(batch_size, envs.obs_shape['observation'][0]).to(policy.actor.device)


    act_Tsor = poli_inf.forward(obs_Tsor)
    print(act_Tsor.shape)

    print("convert policy to onnx format...", end = "")
    torch.onnx.export(model = poli_inf, args = obs_Tsor, f = "walker-stand-ppo-Thu Mar 30 22_03_44 2023-1000.onnx", 
                    export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                'output' : {0 : 'batch_size'}})
    print("succeed!")

def infer_onnx():
    import onnx 
    import onnxruntime

    filename = "walker-stand-ppo-Thu Mar 30 22_03_44 2023-1000.onnx"
    onnx_model = onnx.load(filename)
    onnx.checker.check_model(onnx_model)

    # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}

    # ort_outs = ort_session.run(None, ort_inputs)
    nenvs = 4 # Parallel
    device = "cuda:0" #"cpu"
    test_episode = 1000
    # config = get_config("./config/ppo/", "mujoco")
    envs = [BasicWrapper(DMControl("walker","stand",200)) for i in range(nenvs)]
    # envs = [BasicWrapper(DMControl("walker","stand",200)) for i in range(nenvs)]
    envs = DummyVecEnv(envs)

    poli_inf_ort = onnxruntime.InferenceSession(filename)

    obs,infos = envs.reset() # (nenvs, 24)
    current_episode = 0
    while current_episode < test_episode:
        print("[%03d]"%(current_episode))
        envs.render("human")
        
        ort_inputs = {poli_inf_ort.get_inputs()[0].name: obs['observation'].astype(np.float32)}
        ort_outs = poli_inf_ort.run(None, ort_inputs)[0]

        next_obs,rewards,terminals,trunctions,infos = envs.step(ort_outs)
        for i in range(nenvs):
            if terminals[i] == True or trunctions[i] == True: current_episode += 1
        obs = next_obs


if __name__ == '__main__':

    # convert_onnx()
    infer_onnx()