import redis
import pickle
import gymnasium as gym
import numpy as np
import torch
import threading
from collections import deque
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# Redis client for subscribing to the data
redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

def push_to_redis(channel, data):
    serialized_data = pickle.dumps(data)
    redis_client.publish(channel, serialized_data)


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


class QNetwork(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, 8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, 4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, stride=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(3136, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, env.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def listen_for_model_updates(q_network, pubsub):
    """A function to listen for model updates from Redis in a separate thread."""
    for message in pubsub.listen():
        if message['type'] == 'message':
            model_data = pickle.loads(message['data'])
            q_network.load_state_dict(model_data)

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


def collector(env, epsilon_start, epsilon_end, total_timesteps, device):
    q_network = QNetwork(env).to(device)
    pubsub = redis_client.pubsub()
    pubsub.subscribe('model_update')

    # Start a separate thread to listen for model updates from Redis
    threading.Thread(target=listen_for_model_updates, args=(q_network, pubsub), daemon=True).start()

    obs, _ = env.reset()

    # Buffer for collecting data in batches
    batch_size = 1
    collected_data = deque(maxlen=batch_size)
    episode_count = 0

    for step in range(total_timesteps):
        # Linearly decaying epsilon
        epsilon = linear_epsilon_schedule(epsilon_start, epsilon_end, total_timesteps, step)
        # Epsilon-greedy action selection
        if np.random.rand() < epsilon:
            actions = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])  # Random action
        else:
            with torch.no_grad():
                q_values = q_network(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()  # Choose action with highest Q-value

        next_obs, rewards, terminations, truncations, infos = env.step(actions)
        obs = next_obs

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episode_count += 1  # Increment episode count
                    print(f"Episode {episode_count}, episodic_return={info['episode']['r']}")

        # Collect experience in batch
        collected_data.append({
            'observations': obs,
            'next_observations': next_obs,
            'actions': actions,
            'rewards': rewards,
            'dones': terminations,
        })
        # If batch is full, push to Redis
        if len(collected_data) == batch_size:
            push_to_redis('env_data', list(collected_data))
            collected_data.clear()  # Clear collected data after pushing


# Device setup and environment creation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gym.vector.SyncVectorEnv([make_env("Pong-v4", 1, 0, False, "Pong")])

# Parameters for epsilon decay
epsilon_start = 1.0  # Starting epsilon
epsilon_end = 0.1    # Ending epsilon
total_timesteps = 2000000  # Total number of training steps

# Start collector
# Just call the collector function without threading
collector(env, epsilon_start, epsilon_end, total_timesteps, device)
