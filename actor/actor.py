import pickle
import random

import redis
import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# Redis client for model synchronization and experience publishing
redis_client = redis.StrictRedis(host='redis', port=6379, db=0)
PUBSUB_CHANNEL = "experience_channel"

# Model keys in Redis
MODEL_KEY = "latest_model"
VERSION_KEY = "model_version"


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


class ExperienceBatch:
    """Class to handle batching of experiences before sending to Redis."""

    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.experiences = []

    def add(self, obs, next_obs, action, reward, done, info):
        """Add a single experience to the batch."""
        experience = {
            "obs": obs,
            "next_obs": next_obs,
            "action": action,
            "reward": reward,
            "done": done,
            "info": info
        }
        self.experiences.append(experience)

    def is_ready(self):
        """Check if the batch is ready to be sent."""
        return len(self.experiences) >= self.batch_size

    def get_size(self):
        """Get the current size of the batch."""
        return len(self.experiences)

    def clear(self):
        """Clear the batch after sending."""
        self.experiences = []

    def publish(self):
        """Publish the batch to Redis and clear the batch."""
        if not self.experiences:
            return False

        try:
            # Publish the batch of experiences
            redis_client.publish(PUBSUB_CHANNEL, pickle.dumps(self.experiences))
            self.clear()
            return True
        except Exception as e:
            return False


def make_env(env_id, seed):
    """Create and wrap the environment."""

    def thunk():
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear interpolation schedule for epsilon."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def load_model_from_redis(q_network, current_version):
    """Load model from Redis if a newer version is available."""
    try:
        redis_version = redis_client.get(VERSION_KEY)
        if redis_version is None:
            return current_version, False

        redis_version = int(redis_version)
        if redis_version <= current_version:
            return current_version, False

        model_data = redis_client.get(MODEL_KEY)
        if model_data is None:
            return current_version, False

        q_network.load_state_dict(pickle.loads(model_data))
        return redis_version, True

    except Exception:
        return current_version, False


def actor_loop(env_id, seed, total_steps, batch_size=32, model_sync_frequency=100,
               start_epsilon=1.0, end_epsilon=0.01, epsilon_duration_fraction=0.05,
               force_batch_send_frequency=1000):
    # Set device (actors can run on CPU to save GPU for trainer)
    device = torch.device("cuda")  # Change to "cuda" if you want actors to use GPU

    # Create environment
    env = gym.vector.SyncVectorEnv([make_env(env_id, seed)])

    # Create model
    q_network = QNetwork(env).to(device)

    # Try to load initial model
    model_version = 0
    load_model_from_redis(q_network, model_version)

    # Initialize environment
    obs, _ = env.reset(seed=seed)

    # Create experience batch
    experience_batch = ExperienceBatch(batch_size=batch_size)

    # Main loop variables
    steps = 0
    steps_since_last_send = 0

    # Calculate epsilon decay duration
    epsilon_duration = int(total_steps * epsilon_duration_fraction)

    try:
        while steps < total_steps:
            # Check if it's time to sync model
            if steps % model_sync_frequency == 0:
                model_version, _ = load_model_from_redis(q_network, model_version)

            # Epsilon-greedy action selection
            epsilon = linear_schedule(start_epsilon, end_epsilon, epsilon_duration, steps)
            if random.random() < epsilon:
                actions = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])
            else:
                with torch.no_grad():
                    q_values = q_network(torch.Tensor(obs).to(device))
                    actions = torch.argmax(q_values, dim=1).cpu().numpy()

            # Take action in environment
            next_obs, rewards, terminations, truncations, infos = env.step(actions)

            # Handle episode termination
            dones = terminations | truncations

            # Store transitions
            real_next_obs = next_obs.copy()
            for idx, d in enumerate(dones):
                if d and "final_observation" in infos:
                    real_next_obs[idx] = infos["final_observation"][idx]

            # Add experience to batch
            experience_batch.add(
                obs=obs[0],
                next_obs=real_next_obs[0],
                action=actions[0],
                reward=rewards[0],
                done=dones[0],
                info={}  # We don't need to send all info data
            )

            steps_since_last_send += 1

            # Check if it's time to send the batch
            if experience_batch.is_ready() or steps_since_last_send >= force_batch_send_frequency:
                if experience_batch.publish():
                    steps_since_last_send = 0

            # Update episode counter
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={steps}, episodic_return={info['episode']['r']}")

            # Update for next iteration
            obs = next_obs
            steps += 1

    except KeyboardInterrupt:
        pass
    finally:
        # Send any remaining experiences
        if experience_batch.get_size() > 0:
            experience_batch.publish()
        env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RL Actor for distributed training")
    parser.add_argument("--env-id", type=str, required=False, default="Pong-v4", help="Environment ID")
    parser.add_argument("--actor-id", type=int, required=False, default=0, help="Unique actor ID")
    parser.add_argument("--seed", type=int, required=False, default=None, help="Random seed")
    parser.add_argument("--steps", type=int, required=False, default=10000000, help="Total steps to run")
    parser.add_argument("--batch-size", type=int, required=False, default=32, help="Batch size for experiences")
    parser.add_argument("--sync-freq", type=int, required=False, default=100, help="Model sync frequency")

    args = parser.parse_args()

    # Use actor_id as seed if not provided
    if args.seed is None:
        args.seed = args.actor_id

    actor_loop(
        env_id=args.env_id,
        seed=args.seed,
        total_steps=args.steps,
        batch_size=args.batch_size,
        model_sync_frequency=args.sync_freq
    )
