import redis
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
import datetime

# Redis client for subscribing to the data
redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

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


# Training loop with experience pulled from Redis
def training_loop(env, total_timesteps, learning_starts, train_frequency, batch_size, gamma, device,
                  save_interval, target_network_frequency, tau, model_push_frequency, timestamp_file_path):
    q_network = QNetwork(env).to(device)
    target_network = QNetwork(env).to(device)  # 目标网络
    target_network.load_state_dict(q_network.state_dict())  # 初始化目标网络

    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)

    rb = ReplayBuffer(100000, env.single_observation_space, env.single_action_space, device,
                      optimize_memory_usage=True, handle_timeout_termination=False)

    writer = SummaryWriter('runs/pong_train')
    obs, _ = env.reset()

    global_step = 0
    pubsub = redis_client.pubsub()
    pubsub.subscribe('env_data')

    # Open the timestamp file for writing at the start
    with open(timestamp_file_path, 'w') as timestamp_file:
        timestamp_file.write(f"Training started at: {datetime.datetime.now()}\n")

    while global_step < total_timesteps:
        for message in pubsub.listen():
            # Pull experience data from Redis
            if message['type'] == 'message':
                global_step += 1
                if global_step > total_timesteps:
                    break

                data = pickle.loads(message['data'])
                observations = data['observations']
                next_observations = data['next_observations']
                actions = data['actions']
                rewards = data['rewards']
                dones = data['dones']

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

                # Save the model periodically
                    if global_step % save_interval == 0:
                        model_path = f"runs/pong_model_{global_step}.pth"
                        torch.save(q_network.state_dict(), model_path)
                        print(f"Model saved at step {global_step} to {model_path}")

                        # Log timestamp after model save
                        with open(timestamp_file_path, 'a') as timestamp_file:
                            current_time = datetime.datetime.now()
                            timestamp_file.write(f"Step {global_step}: {current_time}\n")

                    # Update target network periodically (soft update)
                    if global_step % target_network_frequency == 0:
                        print("Update target network. Global step:", global_step)
                        for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                            target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)

                # Periodically push the model to Redis (publisher)
                    if global_step % model_push_frequency == 0:
                        model_data = q_network.state_dict()
                        redis_client.publish('model_update', pickle.dumps(model_data))
                        print(f"Model pushed to Redis at step {global_step}")

    writer.close()


# Device setup and environment creation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
env = gym.vector.SyncVectorEnv([make_env("Pong-v4", 1, 0, False, "Pong")])

# Timestamp file path
timestamp_file_path = "runs/timestamp_log.txt"

# Start training loop
training_loop(env=env, total_timesteps=2000000, learning_starts=100000, train_frequency=4, batch_size=32,
              gamma=0.99, device=device, save_interval=100000, target_network_frequency=1000, tau=0.005,
              model_push_frequency=100, timestamp_file_path=timestamp_file_path)
