import os
import torch
import torch.nn as nn 
import gymnasium as gym
import panda_gym

from huggingface_sb3 import load_from_hub, package_to_hub

from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env

from huggingface_hub import notebook_login

env_id = "PandaReachDense-v3"

# Create the env
env = gym.make(env_id)

# Get the state space and action space
s_size = env.observation_space.shape
a_size = env.action_space

#Initialize the Enviroment 
env = make_vec_env(env_id, n_envs=4)

#Vectorizing & Normalizing Enviroment
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

#Deep Neural Network
policy_kwargs = dict(activation_fn=nn.LeakyReLU,
                     net_arch=dict(pi=[128, 128], vf=[128, 128]))

#Model parameters
model = A2C(policy = "MultiInputPolicy",
            ent_coef=0.55,
            learning_rate=0.0004,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            env = env,
            verbose=1)
#Agent Learning for 1 Million TimeSteps
model.learn(42400)

model.save("A2C-PandaReachDense-v3Xenjin450")
env.save("pandareachdensev3.pkl")