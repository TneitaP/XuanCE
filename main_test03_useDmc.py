import numpy as np 
import mujoco
from dm_control import suite
from dm_control import viewer
import cv2 
import onnx
import onnxruntime

class Policy:
    def __init__(self, action_spec, poli_filename):
        self.a_spec = action_spec
        onnx_model = onnx.load(filename)
        onnx.checker.check_model(onnx_model)
        self.poli_inf_ort = onnxruntime.InferenceSession(poli_filename)

    def get_random_act(self, obs_Arr):
        # del time_step  # Unused.
        # time_step.observation
        # humanoid: (['joint_angles'(21), 'head_height'(1), 'extremities'(12), 'torso_vertical'(3), 'com_velocity'(3), 'velocity'(27)])
        # walker: (['orientations'(14), 'height'(1), 'velocity'(9)])
        return np.random.uniform(low=self.a_spec.minimum,
                                high=self.a_spec.maximum,
                                size=self.a_spec.shape) # shape (21, )

    def get_constant_act(self, obs_Arr):
        return np.ones_like(self.a_spec.minimum)
    
    def get_ppo_act(self, obs_Arr):
        
        ort_inputs = {self.poli_inf_ort.get_inputs()[0].name: obs_Arr.reshape(1, -1)}
        return self.poli_inf_ort.run(None, ort_inputs)[0][0]

if __name__ == '__main__':

    filename = r"D:\zzm_codes\XuanCE\models\Walker(1)Stand\ppo\walker-stand-ppo-Thu Mar 30 22_03_44 2023-1000.onnx"
    

    env = suite.load('walker', 'stand')
    poli = Policy(env.action_spec(), filename)

    time_step_data = env.reset()
    obs_Arr = np.concatenate([val.reshape(-1) for key, val in time_step_data.observation.items()], dtype=np.float32)
    
    succeed_case_count = 0
    epoch_time = 0
    while True:
        action = poli.get_ppo_act(obs_Arr)
        time_step_data = env.step(action)

        camera0_frame = env.physics.render(camera_id=0, height=480, width=640)
        camera1_frame = env.physics.render(camera_id=1, height=480, width=640)
        # print(camera0.shape)
        cv2.imshow("cam0", camera0_frame[:,:,::-1])
        cv2.imshow("cam1", camera1_frame[:,:,::-1])
        cv2.waitKey(1)

        reward = time_step_data.reward
        if reward is None: reward = 0 
        if abs(reward - 1) < 1e-3:
            print("task succeed! reward = ", reward)
            succeed_case_count +=1
            time_step_data = env.reset()

        if epoch_time > 1000:
            print("time out! reward = ", reward)
            time_step_data = env.reset()
            epoch_time = 0

        next_obs_Arr = np.concatenate([val.reshape(-1) for key, val in time_step_data.observation.items()], dtype=np.float32)
        obs_Arr = next_obs_Arr
        epoch_time +=1 
