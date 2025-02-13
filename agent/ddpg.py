from agent import *
class DDPG_Agent:
    def __init__(self,
                 config,
                 environment,
                 policy,
                 learner):
        self.config = config
        self.environment = environment
        self.policy = policy
        self.learner = learner
        self.nenvs = environment.num_envs
        self.nsize = config.nsize
        self.minibatch = config.minibatch
        self.gamma = config.gamma
        
        self.input_shape = self.policy.input_shape
        self.action_shape = self.environment.action_space.shape
        self.output_shape = self.policy.output_shape
        
        self.start_noise = config.start_noise 
        self.end_noise = config.end_noise
        self.noise = self.start_noise

        self.start_training_size = config.start_training_size
        self.training_frequency = config.training_frequency
        self.memory = DummyOffPolicyBuffer(self.input_shape,
                                           self.action_shape,
                                           self.output_shape,
                                           self.nenvs,
                                           self.nsize,
                                           self.minibatch)
        self.summary = SummaryWriter(self.config.logdir)
        
        self.train_episodes = np.zeros((self.nenvs,),np.int32)
        self.train_steps = 0
    
    def interact(self,inputs,noise):
        outputs,action,_ = self.policy(inputs)
        action = action.detach().cpu().numpy()
        action = action + np.random.normal(size=action.shape)*noise
        for key,value in zip(outputs.keys(),outputs.values()):
            outputs[key] = value.detach().cpu().numpy()
        return outputs,np.clip(action,-1,1)
    
    def train(self,train_steps:int=10000):
        obs,infos = self.environment.reset()
        for _ in tqdm(range(train_steps)):
            outputs,actions = self.interact(obs,self.noise)
            if self.train_steps < self.config.start_training_size:
                actions = [self.environment.action_space.sample() for i in range(self.nenvs)]
            next_obs,rewards,terminals,trunctions,infos = self.environment.step(actions)
            store_next_obs = next_obs.copy()
            for i in range(self.nenvs):
                if trunctions[i]:
                    for key in infos[i].keys():
                        if key in store_next_obs.keys():
                            store_next_obs[key][i] = infos[i][key]
            self.memory.store(obs,actions,outputs,rewards,terminals,store_next_obs)
            if self.memory.size >= self.start_training_size and self.train_steps % self.training_frequency == 0:
                input_batch,action_batch,output_batch,reward_batch,terminal_batch,next_input_batch = self.memory.sample()
                self.learner.update(input_batch,action_batch,reward_batch,terminal_batch,next_input_batch)
            for i in range(self.nenvs):
                if terminals[i] or trunctions[i]:
                    self.train_episodes[i] += 1
                    self.summary.add_scalars("rewards-episode",{"env-%d"%i:infos[i]['episode_score']},self.train_episodes[i])
                    self.summary.add_scalars("rewards-steps",{"env-%d"%i:infos[i]['episode_score']},self.train_steps)
            obs = next_obs
            self.train_steps += 1
            self.noise = self.noise - (self.start_noise-self.end_noise)/self.config.train_steps
            
    def test(self,test_episode=10,render=False):
        import copy
        test_environment = copy.deepcopy(self.environment)
        obs,infos = test_environment.reset()
        current_episode = 0
        scores = []
        while current_episode < test_episode:
            if render:
                test_environment.render("human")
            outputs,actions = self.interact(obs,0)
            next_obs,rewards,terminals,trunctions,infos = test_environment.step(actions)
            for i in range(self.nenvs):
                if terminals[i] == True or trunctions[i] == True:
                    scores.append(infos[i]['episode_score'])
                    current_episode += 1
            obs = next_obs
        return scores
    
    def benchmark(self,train_steps:int=10000,evaluate_steps:int=10000,test_episode=10,render=False):
        import time
        epoch = int(train_steps / evaluate_steps)
        benchmark_scores = []
        benchmark_scores.append({'steps':self.train_steps,'scores':self.test(test_episode,render)})
        for i in range(epoch):
            if i == epoch - 1:
                train_step = train_step - ((i+1)*evaluate_steps)
            else:
                train_step = evaluate_steps
            print("benchmark epoch[%03d | %03d]"%(i, epoch))
            self.train(train_step)
            benchmark_scores.append({'steps':self.train_steps,'scores':self.test(test_episode,render)})
        time_string = get_time_str()
        np.save(self.config.logdir+"benchmark_%s.npy"%time_string, benchmark_scores)
            
        
             