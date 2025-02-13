from math import trunc
from environment import *
from environment.vectorize import VecEnv
class RewardNorm(VecEnv):
    def __init__(self,config,vecenv:VecEnv,scale_range=(0.1,10),reward_range=(-5,5),gamma=0.99,train=True,pretrain_para_path = None):
        super(RewardNorm,self).__init__(vecenv.num_envs,vecenv.observation_space,vecenv.action_space)
        assert scale_range[0] < scale_range[1], "invalid scale_range."
        assert reward_range[0] < reward_range[1], "Invalid reward_range."
        assert gamma < 1, "Gamma should be a float value smaller than 1."
        self.config = config
        self.gamma = gamma
        self.scale_range = scale_range
        self.reward_range = reward_range
        self.vecenv = vecenv
        self.return_rms = Running_MeanStd({'return':(1,)})
        self.episode_rewards = [[] for i in range(self.num_envs)]
        self.train_steps = 0
        if pretrain_para_path is not None:
            self.load_rms(pretrain_para_path)
        elif train == False:
            self.load_rms(os.path.join(self.config.modeldir,"reward_stat.npy"))
        

    def load_rms(self, npy_path):
        rms_data = np.load(npy_path,allow_pickle=True).item()
        self.return_rms.count = rms_data['count']
        self.return_rms.mean = rms_data['mean']
        self.return_rms.var = rms_data['var']
             
    def step_wait(self):
        obs,rews,terminals,trunctions,infos = self.vecenv.step_wait()
        for i in range(len(rews)):
            if terminals[i] != True and trunctions[i] != True:
                self.episode_rewards[i].append(rews[i])
            else:
                self.return_rms.update({'return':discount_cumsum(self.episode_rewards[i],self.gamma)[0:1][np.newaxis,:]})
                self.episode_rewards[i].clear()
            scale = np.clip(self.return_rms.std['return'][0],self.scale_range[0],self.scale_range[1])
            rews[i] =  np.clip(rews[i]/scale,self.reward_range[0],self.reward_range[1])
        return obs,rews,terminals,trunctions,infos
    def reset(self):
        return self.vecenv.reset()
    def step_async(self, actions):
        self.train_steps += 1
        if self.config.train_steps == self.train_steps or self.train_steps % self.config.save_model_frequency == 0:
            np.save(os.path.join(self.config.modeldir,"reward_stat.npy"),{'count':self.return_rms.count,'mean':self.return_rms.mean,'var':self.return_rms.var})
        return self.vecenv.step_async(actions)
    def get_images(self):
        return self.vecenv.get_images()
    def close_extras(self):
        return self.vecenv.close_extras()

class ObservationNorm(VecEnv):
    def __init__(self,config,vecenv:VecEnv,scale_range=(0.1,10),obs_range=(-5,5),forbidden_keys=[],train=True,pretrain_para_path = None):
        super(ObservationNorm,self).__init__(vecenv.num_envs,vecenv.observation_space,vecenv.action_space)
        assert scale_range[0] < scale_range[1], "invalid scale_range."
        assert obs_range[0] < obs_range[1], "Invalid reward_range."
        self.config = config
        self.scale_range = scale_range
        self.obs_range = obs_range
        self.forbidden_keys = forbidden_keys
        self.vecenv = vecenv
        self.obs_rms = Running_MeanStd(space2shape(vecenv.observation_space))
        self.train_steps = 0
        if pretrain_para_path is not None:
            self.load_rms(pretrain_para_path)
        elif train == False:
            self.load_rms(os.path.join(self.config.modeldir,"observation_stat.npy"))
    
    def load_rms(self, npy_path):
        rms_data = np.load(npy_path,allow_pickle=True).item()
        self.obs_rms.count = rms_data['count']
        self.obs_rms.mean = rms_data['mean']
        self.obs_rms.var = rms_data['var']
    def step_wait(self):
        obs,rews,terminals,trunctions,infos = self.vecenv.step_wait()
        self.obs_rms.update(obs)
        norm_observation = {}
        for key,value in zip(obs.keys(),obs.values()):
            if key in self.forbidden_keys:
                continue
            scale_factor = np.clip(1/(self.obs_rms.std[key] + 1e-7),self.scale_range[0],self.scale_range[1])
            norm_observation[key] = np.clip((value - self.obs_rms.mean[key]) * scale_factor,self.obs_range[0],self.obs_range[1])
        return norm_observation,rews,terminals,trunctions,infos
    def reset(self):
        return self.vecenv.reset()
    def step_async(self, actions):
        self.train_steps += 1
        if self.config.train_steps == self.train_steps or self.train_steps % self.config.save_model_frequency == 0:
            np.save(os.path.join(self.config.modeldir,"observation_stat.npy"),{'count':self.obs_rms.count,'mean':self.obs_rms.mean,'var':self.obs_rms.var})
        return self.vecenv.step_async(actions)
    def get_images(self):
        return self.vecenv.get_images()
    def close_extras(self):
        return self.vecenv.close_extras()