import pickle
import time
import datetime
import os
import json

import redis
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import random
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from torch.utils.tensorboard import SummaryWriter

# Redis client for communication
redis_client = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)


def make_env(env_id, seed, idx, capture_video, run_dir):
    def thunk():
        if capture_video and idx == 0:
            video_dir = os.path.join(run_dir, "videos")
            os.makedirs(video_dir, exist_ok=True)
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, video_dir)
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


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


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


class ModelSynchronizer:
    def __init__(self, model_key="latest_model", version_key="model_version"):
        self.model_key = model_key
        self.version_key = version_key
        self.redis_client = redis.StrictRedis(host='127.0.0.1', port=6379, db=0)

    def save_model(self, model, version):
        """ Save the latest model to Redis with a version number only if the version is newer. """
        redis_version = self.redis_client.get(self.version_key)
        if redis_version is not None:
            redis_version = int(redis_version)
            if version > redis_version:
                # Only save the model if the local version is greater than the Redis version
                model_data = pickle.dumps(model.state_dict())
                self.redis_client.set(self.model_key, model_data)
                self.redis_client.set(self.version_key, version)  # Save the model version
                print(f"Model saved to Redis with version {version}.")
            else:
                print(f"Local version {version} is not newer than Redis version {redis_version}. Model not saved.")
        else:
            # If Redis doesn't have a version, save the model
            model_data = pickle.dumps(model.state_dict())
            self.redis_client.set(self.model_key, model_data)
            self.redis_client.set(self.version_key, version)  # Save the model version
            print(f"Model saved to Redis with version {version}.")

    def load_model(self, model, local_version):
        """ Load the latest model from Redis only if the version is newer. """
        redis_version = self.redis_client.get(self.version_key)
        if redis_version is not None:
            redis_version = int(redis_version)
            if redis_version > local_version:
                model_data = self.redis_client.get(self.model_key)
                if model_data is not None:
                    model.load_state_dict(pickle.loads(model_data))
                    print(f"Loaded model from Redis with version {redis_version}.")
                    return redis_version
                else:
                    print("No model found in Redis.")
                    return local_version
            else:
                return local_version
        else:
            print("No model found in Redis, using local model.")
            return local_version


def evaluate(model, env_id, eval_episodes, run_dir, device, epsilon=0.05, capture_video=True,
             evaluation_name="evaluation"):
    """
    Evaluate the performance of a model for a given number of episodes.

    Args:
        model: The trained model to evaluate
        env_id: Environment ID
        eval_episodes: Number of episodes to evaluate
        run_dir: Directory for saving evaluation results
        device: Device to run the model on
        epsilon: Epsilon value for epsilon-greedy policy during evaluation
        capture_video: Whether to capture video during evaluation
        evaluation_name: Name of this evaluation run

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Starting {evaluation_name} for {eval_episodes} episodes with epsilon={epsilon}")

    # Create evaluation environment
    eval_env = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_dir)])

    # Set model to evaluation mode
    model.eval()

    obs, _ = eval_env.reset()
    episodic_returns = []

    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array([eval_env.single_action_space.sample() for _ in range(eval_env.num_envs)])
        else:
            with torch.no_grad():
                q_values = model(torch.Tensor(obs).to(device))
                actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, _, _, _, infos = eval_env.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if "episode" not in info:
                    continue
                print(f"{evaluation_name}_episode={len(episodic_returns)}, episodic_return={info['episode']['r']}")
                episodic_returns.append(info["episode"]["r"])

        obs = next_obs

    eval_env.close()

    # Calculate statistics
    mean_return = np.mean(episodic_returns)
    median_return = np.median(episodic_returns)
    min_return = np.min(episodic_returns)
    max_return = np.max(episodic_returns)

    print(f"{evaluation_name.capitalize()} results:")
    print(f"  Mean return: {mean_return:.2f}")
    print(f"  Median return: {median_return:.2f}")
    print(f"  Min return: {min_return:.2f}")
    print(f"  Max return: {max_return:.2f}")

    # Create evaluation metrics
    evaluation_metrics = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "mean_return": float(mean_return),
        "median_return": float(median_return),
        "min_return": float(min_return),
        "max_return": float(max_return),
        "returns": [float(r) for r in episodic_returns]
    }

    return evaluation_metrics


def save_model_checkpoint(model, run_dir, checkpoint_name):
    """
    Save model checkpoint to a specific location.

    Args:
        model: Model to save
        run_dir: Run directory
        checkpoint_name: Name of the checkpoint
    """
    # Ensure models directory exists
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    # Save the model
    checkpoint_path = os.path.join(models_dir, f"{checkpoint_name}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")

    return checkpoint_path


def save_evaluation_results(metrics, run_dir, evaluation_name):
    """
    Save evaluation results to a JSON file.

    Args:
        metrics: Dictionary with evaluation metrics
        run_dir: Run directory
        evaluation_name: Name of this evaluation
    """
    # Ensure evaluations directory exists
    eval_dir = os.path.join(run_dir, "evaluations")
    os.makedirs(eval_dir, exist_ok=True)

    # Save metrics to JSON
    metrics_path = os.path.join(eval_dir, f"{evaluation_name}.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"Evaluation results saved to {metrics_path}")

    return metrics_path


def setup_run_directory(env_id):
    """Set up a run directory with a timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{env_id}_{timestamp}"
    run_dir = os.path.join("runs", run_name)

    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "evaluations"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    return run_dir


def save_training_config(config, run_dir):
    """Save training configuration to a JSON file"""
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Training configuration saved to {config_path}")
    return config_path


def training_loop(env, env_id, total_timesteps, learning_starts, train_frequency, batch_size, gamma, device,
                  target_network_frequency, tau, eval_frequency=100000, checkpoint_frequency=100000):
    """
    Main training loop for DQN with periodic evaluations and checkpoints.

    Args:
        env: Environment
        env_id: Environment ID
        total_timesteps: Total timesteps for training
        learning_starts: Learning starts after this many steps
        train_frequency: Train every N steps
        batch_size: Batch size for training
        gamma: Discount factor
        device: Device to run on (cuda/cpu)
        target_network_frequency: Update target network every N steps
        tau: Soft update coefficient
        eval_frequency: Evaluate model every N steps
        checkpoint_frequency: Save model checkpoint every N steps
    """
    # Setup run directory with timestamp
    run_dir = setup_run_directory(env_id)

    # Save training configuration
    config = {
        "env_id": env_id,
        "total_timesteps": total_timesteps,
        "learning_starts": learning_starts,
        "train_frequency": train_frequency,
        "batch_size": batch_size,
        "gamma": gamma,
        "device": str(device),
        "target_network_frequency": target_network_frequency,
        "tau": tau,
        "eval_frequency": eval_frequency,
        "checkpoint_frequency": checkpoint_frequency,
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_training_config(config, run_dir)

    # Setup tensorboard writer
    writer = SummaryWriter(os.path.join(run_dir, "logs"))

    # Initialize networks
    q_network = QNetwork(env).to(device)
    target_network = QNetwork(env).to(device)

    # Start with local version 0
    local_version = 0
    model_synchronizer = ModelSynchronizer()

    # Check if we need to load a newer model version from Redis
    local_version = model_synchronizer.load_model(target_network, local_version)
    q_network.load_state_dict(target_network.state_dict())

    # Save initial model
    save_model_checkpoint(q_network, run_dir, "initial_model")

    # Setup optimizer
    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)

    # Setup replay buffer
    rb = ReplayBuffer(
        200000,
        env.single_observation_space,
        env.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False
    )

    start_time = time.time()
    obs, _ = env.reset()

    # Store evaluation results
    evaluations = []

    # Main training loop
    for global_step in range(total_timesteps):
        # Epsilon-greedy action selection (for exploration)
        epsilon = linear_schedule(1.0, 0.01, 0.1 * total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([env.single_action_space.sample() for _ in range(env.num_envs)])  # Random action
        else:
            q_values = q_network(torch.Tensor(obs).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()  # Choose action with highest Q-value

        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        # Log episode metrics
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    writer.add_scalar("charts/epsilon", epsilon, global_step)

        # Handle truncated episodes
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        # Add to replay buffer
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs

        # Training step
        if global_step > learning_starts and global_step % train_frequency == 0:
            data = rb.sample(batch_size)

            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())

            old_val = q_network(data.observations).gather(1, data.actions).squeeze()
            loss = F.mse_loss(td_target, old_val)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("charts/loss", loss.item(), global_step)

            # Log training metrics periodically
            if global_step % 100 == 0:
                sps = int(global_step / (time.time() - start_time))
                print(f"Step: {global_step}/{total_timesteps}, SPS: {sps}, Loss: {loss.item():.5f}")
                writer.add_scalar("charts/SPS", sps, global_step)

                # Try to load newer model from Redis
                local_version = model_synchronizer.load_model(target_network, local_version)

            # Update target network periodically
            if global_step % target_network_frequency == 0:
                for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)

                # Update the model version and save it to Redis
                local_version += 1
                model_synchronizer.save_model(target_network, local_version)

                # Log target network update
                writer.add_scalar("charts/target_updates", local_version, global_step)

        # Periodic evaluation
        if global_step > 0 and global_step % eval_frequency == 0:
            # Run evaluation
            eval_metrics = evaluate(
                model=target_network,
                env_id=env_id,
                eval_episodes=10,
                run_dir=run_dir,
                device=device,
                epsilon=0.05,
                capture_video=(global_step % (eval_frequency * 5) == 0),  # Record video less frequently
                evaluation_name=f"eval_{global_step // 1000}k"
            )

            # Add current step information
            eval_metrics["step"] = global_step

            # Save evaluation results
            save_evaluation_results(eval_metrics, run_dir, f"eval_{global_step // 1000}k")

            # Add to evaluations list
            evaluations.append(eval_metrics)

            # Log to tensorboard
            writer.add_scalar("evaluation/mean_return", eval_metrics["mean_return"], global_step)
            writer.add_scalar("evaluation/median_return", eval_metrics["median_return"], global_step)

        # Periodic checkpoint saving
        if global_step > 0 and global_step % checkpoint_frequency == 0:
            save_model_checkpoint(target_network, run_dir, f"checkpoint_{global_step // 1000}k")

    print(f"Training completed! Total time: {time.time() - start_time:.2f} seconds")

    # Save all evaluations in a single file
    all_evals_path = os.path.join(run_dir, "evaluations", "Distributed_2Actor.json.json")
    with open(all_evals_path, 'w') as f:
        json.dump(evaluations, f, indent=4)

    # Final model saving
    final_model_path = save_model_checkpoint(target_network, run_dir, "final_model")

    # Final evaluation
    print("Running final evaluation...")
    final_eval_metrics = evaluate(
        model=target_network,
        env_id=env_id,
        eval_episodes=20,  # More episodes for final evaluation
        run_dir=run_dir,
        device=device,
        epsilon=0.05,
        capture_video=True,
        evaluation_name="final_evaluation"
    )

    # Add final information
    final_eval_metrics["step"] = total_timesteps
    final_eval_metrics["total_training_time"] = time.time() - start_time

    # Save final evaluation
    save_evaluation_results(final_eval_metrics, run_dir, "final_evaluation")

    # Close tensorboard writer
    writer.close()

    print(f"All training results saved to {run_dir}")
    print(f"Final model saved to {final_model_path}")

    return {
        "run_dir": run_dir,
        "final_model_path": final_model_path,
        "final_eval_metrics": final_eval_metrics
    }


if __name__ == "__main__":
    # Setup for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id = "Pong-v4"
    env = gym.vector.SyncVectorEnv([make_env(env_id, 1, 0, False, "temp")])  # temp dir will be replaced

    # Start training
    results = training_loop(
        env=env,
        env_id=env_id,
        total_timesteps=1000000,
        learning_starts=50000,
        train_frequency=4,
        batch_size=32,
        gamma=0.99,
        device=device,
        target_network_frequency=1000,
        tau=1.0,
        eval_frequency=100000,
        checkpoint_frequency=100000
    )

    print(f"Training successfully completed!")
    print(f"Run directory: {results['run_dir']}")
    print(f"Final model: {results['final_model_path']}")
    print(f"Final mean return: {results['final_eval_metrics']['mean_return']:.2f}")