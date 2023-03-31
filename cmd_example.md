activate handPhy23

## Training with pre-trained weights, rwd_norm and obs_norm
python main.py --render True --pretrain_weight "D:\zzm_codes\XuanCE\_old0329\models\Walker(1)\ppo\model-Wed Mar 29 21_18_55 2023-18000.pth" --pretrain_reward_norm "D:\zzm_codes\XuanCE\_old0329\models\Walker(1)\ppo\reward_stat.npy" --pretrain_observe_norm "D:\zzm_codes\XuanCE\_old0329\models\Walker(1)\ppo\observation_stat.npy"


## Enable Tensorboard
tensorboard --logdir logs/Walker(1)Stand/ppo --port 8890
