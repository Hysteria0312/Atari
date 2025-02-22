import os
import random

import numpy as np
import torch
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import NoopResetEnv, MaxAndSkipEnv, EpisodicLifeEnv, FireResetEnv, \
    ClipRewardEnv
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)
def load_model(model_path, env, device):
    q_network = QNetwork(env).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()
    return q_network


# 创建环境
def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)

        env.action_space.seed(seed)
        return env

    return thunk


# 生成视频
def generate_video(model_path, env_id, run_name, device, eval_episodes=1, epsilon=0.05):
    env = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, True, run_name)])

    model = load_model(model_path, env, device)

    obs, _ = env.reset()

    for _ in range(eval_episodes):
        done = False
        while not done:
            if random.random() < epsilon:
                actions = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])
            else:
                q_values = model(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

            next_obs, _, terminations, truncations, infos = env.step(actions)

            done = any(terminations) or any(truncations)

            obs = next_obs

    env.close()


if __name__ == "__main__":
    model_path = "./pong_model_100000.pth"
    env_id = "Pong-v4"
    run_name = "replay"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generate_video(model_path, env_id, run_name, device)
