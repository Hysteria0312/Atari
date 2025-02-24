import os
import pickle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from random import random
import redis
import datetime
from typing import Callable
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
# Redis client for subscribing to the data
redis_client = redis.StrictRedis(host='redis', port=6379, db=0)


def evaluate(
        model_path: str,
        make_env: Callable,
        env_id: str,
        eval_episodes: int,
        run_name: str,
        Model: torch.nn.Module,
        device: torch.device = torch.device("cpu"),
        epsilon: float = 0.05,
        capture_video: bool = True,
):
    # Check if model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])
    model = Model(envs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    obs, _ = envs.reset()
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = model(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()
        next_obs, _, _, _, infos = envs.step(actions)
        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"eval_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns += [info["episode"]["r"]]
        obs = next_obs

    return episodic_returns


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


# Create environment with wrappers
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


def linear_epsilon_schedule(epsilon_start, epsilon_end, total_timesteps, current_step):
    """
    Linearly decays epsilon from epsilon_start to epsilon_end.
    :param epsilon_start: Initial epsilon value.
    :param epsilon_end: Final epsilon value (usually a small number, close to 0).
    :param total_timesteps: Total training timesteps.
    :param current_step: Current training step.
    :return: Updated epsilon value.
    """
    epsilon = epsilon_start - (epsilon_start - epsilon_end) * (current_step / total_timesteps)
    return max(epsilon, epsilon_end)  # Ensure epsilon doesn't go below epsilon_end


# Training loop with experience pulled from Redis
def training_loop(env, total_timesteps, learning_starts, train_frequency, batch_size, gamma, device,
                  save_interval, target_network_frequency, tau, model_push_frequency, timestamp_file_path,
                  eval_episodes=10):
    q_network = QNetwork(env).to(device)
    target_network = QNetwork(env).to(device)  # 目标网络
    target_network.load_state_dict(q_network.state_dict())  # 初始化目标网络

    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)

    rb = ReplayBuffer(200000, env.single_observation_space, env.single_action_space, device,
                      optimize_memory_usage=True, handle_timeout_termination=False)

    writer = SummaryWriter('runs/pong_train')
    obs, _ = env.reset()

    global_step = 0
    pubsub = redis_client.pubsub()
    pubsub.subscribe('env_data')

    # Open the timestamp file for writing at the start
    with open(timestamp_file_path, 'w') as timestamp_file:
        timestamp_file.write(f"Training started at: {datetime.datetime.now()}\n")

    episode_returns = []  # To store the returns of each episode

    while global_step < total_timesteps:
        for message in pubsub.listen():
            # Ensure we are receiving messages
            if message['type'] == 'message':
                data = pickle.loads(message['data'])
                # Loop through the batch of experiences
                for experience in data:
                    global_step += 1
                    observations = experience['observations']
                    next_observations = experience['next_observations']
                    actions = experience['actions']
                    rewards = experience['rewards']
                    dones = experience['dones']
                    infos = [{"TimeLimit.truncated": False} for _ in range(env.num_envs)]
                    rb.add(observations, next_observations, actions, rewards, dones, infos)

                if global_step > learning_starts and global_step % train_frequency == 0:
                    batch = rb.sample(batch_size)
                    with torch.no_grad():
                        target_max, _ = target_network(batch.next_observations).max(dim=1)
                        td_target = batch.rewards.flatten() + gamma * target_max * (1 - batch.dones.flatten())
                    old_val = q_network(batch.observations).gather(1, batch.actions).squeeze()
                    loss = F.mse_loss(td_target, old_val)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)

                    # Update target network periodically (soft update)
                    if global_step % target_network_frequency == 0:
                        for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)

                    # Periodically push the model to Redis (publisher)
                    if global_step % model_push_frequency == 0:
                        model_data = q_network.state_dict()
                        redis_client.publish('model_update', pickle.dumps(model_data))
                        print(f"Model pushed to Redis at step {global_step}")

                        # Save the model to the disk
                        model_save_path = f"runs/pong_model_{global_step}.pth"
                        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
                        torch.save(q_network.state_dict(), model_save_path)
                        print(f"Model saved at step {global_step}")

                    # Evaluate every 100,000 steps
                    if global_step % save_interval == 0:
                        print(f"Evaluating model at step {global_step}")
                        episodic_returns = evaluate(
                            model_path=f"runs/pong_model_{global_step}.pth",
                            make_env=make_env,
                            env_id="Pong-v4",
                            eval_episodes=eval_episodes,
                            run_name=f"evaluation_{global_step}",
                            Model=QNetwork,
                            device=device,
                            epsilon=0.05
                        )
                        # Log episodic returns to TensorBoard
                        for idx, episodic_return in enumerate(episodic_returns):
                            writer.add_scalar("eval/episodic_return", episodic_return, global_step + idx)
                        # Log timestamp every 100,000 steps: open, write and close the file immediately
                        current_time = datetime.datetime.now()
                        timestamp_file_path = f"runs/timestamps.txt"
                        with open(timestamp_file_path, 'a') as timestamp_file:
                            timestamp_file.write(f"Step {global_step}: {current_time}\n")
                        print(f"Logged timestamp at step {global_step}: {current_time}")

    writer.close()


# Device setup and environment creation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
env = gym.vector.SyncVectorEnv([make_env("Pong-v4", 1, 0, False, "Pong")])

# Timestamp file path
timestamp_file_path = "runs/timestamp_log.txt"

# Start training loop
training_loop(env=env, total_timesteps=4000000, learning_starts=100000, train_frequency=4, batch_size=32,
              gamma=0.99, device=device, save_interval=100000, target_network_frequency=1000, tau=0.005,
              model_push_frequency=100, timestamp_file_path=timestamp_file_path)
