import pickle
import random
import time
import datetime
import os
import json
import threading
import gymnasium as gym

import redis
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

# Redis client for model synchronization and experience collection
redis_client = redis.StrictRedis(host='redis', port=6379, db=0)
PUBSUB_CHANNEL = "experience_channel"


def make_env(env_id, seed, idx, capture_video, run_dir):
    """Create a wrapped environment for evaluation"""

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


class QNetwork(nn.Module):
    def __init__(self, observation_space, action_space):
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
            nn.Linear(512, action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


class ModelSynchronizer:
    def __init__(self, model_key="latest_model", version_key="model_version"):
        self.model_key = model_key
        self.version_key = version_key
        self.redis_client = redis.StrictRedis(host='redis', port=6379, db=0)

    def save_model(self, model, version):
        """ Save the model to Redis with a version number. """
        model_data = pickle.dumps(model.state_dict())
        self.redis_client.set(self.model_key, model_data)
        self.redis_client.set(self.version_key, version)
        print(f"Model saved to Redis with version {version}.")


class ExperienceCollector:
    def __init__(self, replay_buffer):
        self.replay_buffer = replay_buffer
        self.pubsub = redis_client.pubsub()
        self.pubsub.subscribe(PUBSUB_CHANNEL)
        self.running = True
        self.thread = None

    def start_collecting(self):
        """Start the experience collection thread."""
        self.thread = threading.Thread(target=self._collection_loop)
        self.thread.daemon = True
        self.thread.start()
        print("Experience collection thread started.")

    def stop_collecting(self):
        """Stop the experience collection thread."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        self.pubsub.unsubscribe()
        print("Experience collection stopped.")

    def _collection_loop(self):
        """Main loop for collecting experiences from actors."""
        print("Experience collection loop started.")
        experiences_received = 0

        for message in self.pubsub.listen():
            if not self.running:
                break

            if message["type"] == "message":
                try:
                    # Decode and deserialize the experience data
                    # Now the message contains a batch of experiences rather than a single one
                    experience_batch = pickle.loads(message["data"])

                    batch_size = len(experience_batch)

                    # Process each experience in the batch
                    for experience_data in experience_batch:
                        # Add to replay buffer
                        self.replay_buffer.add(
                            experience_data["obs"],
                            experience_data["next_obs"],
                            experience_data["action"],
                            experience_data["reward"],
                            experience_data["done"],
                            experience_data["info"]
                        )

                    experiences_received += batch_size

                except Exception as e:
                    print(f"Error processing experience batch: {e}")


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
    run_name = f"{env_id}_trainer_{timestamp}"
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


def save_model_checkpoint(model, run_dir, checkpoint_name):
    """Save model checkpoint to a specific location."""
    models_dir = os.path.join(run_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    checkpoint_path = os.path.join(models_dir, f"{checkpoint_name}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Model checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def trainer_loop(observation_space, action_space, env_id, total_timesteps,
                 batch_size, gamma, device, target_network_frequency, tau,
                 checkpoint_frequency=10000, eval_frequency=100000, buffer_size=200000):
    run_dir = setup_run_directory(env_id)

    # Save training configuration
    config = {
        "env_id": env_id,
        "total_timesteps": total_timesteps,
        "batch_size": batch_size,
        "gamma": gamma,
        "device": str(device),
        "target_network_frequency": target_network_frequency,
        "tau": tau,
        "checkpoint_frequency": checkpoint_frequency,
        "eval_frequency": eval_frequency,
        "buffer_size": buffer_size,
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    save_training_config(config, run_dir)

    # Setup tensorboard writer
    writer = SummaryWriter(os.path.join(run_dir, "logs"))

    # Initialize networks
    q_network = QNetwork(observation_space, action_space).to(device)
    target_network = QNetwork(observation_space, action_space).to(device)
    target_network.load_state_dict(q_network.state_dict())  # Initial sync

    # Setup model synchronizer
    model_version = 0
    model_synchronizer = ModelSynchronizer()

    # Save initial model to Redis
    model_synchronizer.save_model(q_network, model_version)

    # Save initial model checkpoint
    save_model_checkpoint(q_network, run_dir, "initial_model")

    # Setup optimizer
    optimizer = optim.Adam(q_network.parameters(), lr=1e-4)

    # Setup replay buffer
    rb = ReplayBuffer(
        buffer_size,
        observation_space,
        action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False
    )

    # Setup experience collector
    collector = ExperienceCollector(rb)
    collector.start_collecting()

    # Main training loop
    start_time = time.time()
    total_updates = 0

    # Store evaluation results
    evaluations = []

    try:
        while total_updates < total_timesteps:
            # Only train if we have enough samples
            if rb.size() < batch_size:
                time.sleep(1)
                continue

            # Sample from replay buffer
            data = rb.sample(batch_size)

            # Calculate TD target
            with torch.no_grad():
                target_max, _ = target_network(data.next_observations).max(dim=1)
                td_target = data.rewards.flatten() + gamma * target_max * (1 - data.dones.flatten())

            # Calculate current Q-values
            current_q = q_network(data.observations).gather(1, data.actions).squeeze()

            # Calculate loss
            loss = F.mse_loss(td_target, current_q)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Increment update counter
            total_updates += 1

            # Log metrics periodically
            if total_updates % 100 == 0:
                sps = int(total_updates / (time.time() - start_time))
                print(
                    f"Updates: {total_updates}/{total_timesteps}, SPS: {sps}")
                writer.add_scalar("charts/loss", loss.item(), total_updates)
                writer.add_scalar("charts/SPS", sps, total_updates)
                model_version += 1
                model_synchronizer.save_model(q_network, model_version)

            # Update target network periodically
            if total_updates % target_network_frequency == 0:
                # Soft update
                for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(tau * q_param.data + (1.0 - tau) * target_param.data)

                # Increment version and save to Redis
                writer.add_scalar("charts/model_version", model_version, total_updates)

            # Periodic evaluation
            if total_updates % eval_frequency == 0:
                # Run evaluation
                eval_metrics = evaluate(
                    model=target_network,
                    env_id=env_id,
                    eval_episodes=10,
                    run_dir=run_dir,
                    device=device,
                    epsilon=0.05,
                    capture_video=(total_updates % (eval_frequency * 5) == 0),  # Record video less frequently
                    evaluation_name=f"eval_{total_updates // 1000}k"
                )

                # Add current step information
                eval_metrics["step"] = total_updates

                # Save evaluation results
                save_evaluation_results(eval_metrics, run_dir, f"eval_{total_updates // 1000}k")

                # Add to evaluations list
                evaluations.append(eval_metrics)

                # Log to tensorboard
                writer.add_scalar("evaluation/mean_return", eval_metrics["mean_return"], total_updates)
                writer.add_scalar("evaluation/median_return", eval_metrics["median_return"], total_updates)

            # Periodic checkpoint saving
            if total_updates % checkpoint_frequency == 0:
                save_model_checkpoint(target_network, run_dir, f"checkpoint_{total_updates // 1000}k")

    except KeyboardInterrupt:
        print("Training interrupted by user")
    finally:
        # Clean up
        collector.stop_collecting()

        # Save final model
        final_model_path = save_model_checkpoint(target_network, run_dir, "final_model")
        model_synchronizer.save_model(target_network, model_version + 1)

        # Save all evaluations in a single file
        all_evals_path = os.path.join(run_dir, "evaluations", "Distributed_2Actor.json.json")
        with open(all_evals_path, 'w') as f:
            json.dump(evaluations, f, indent=4)

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
        final_eval_metrics["step"] = total_updates
        final_eval_metrics["total_training_time"] = time.time() - start_time

        # Save final evaluation
        save_evaluation_results(final_eval_metrics, run_dir, "final_evaluation")

        # Close tensorboard writer
        writer.close()

        print(f"Training completed! Total time: {time.time() - start_time:.2f} seconds")
        print(f"Run directory: {run_dir}")
        print(f"Final model: {final_model_path}")

        return {
            "run_dir": run_dir,
            "final_model_path": final_model_path,
            "model_version": model_version + 1,
            "final_eval_metrics": final_eval_metrics
        }


if __name__ == "__main__":
    from gymnasium.spaces import Box, Discrete
    import argparse

    parser = argparse.ArgumentParser(description="RL Trainer for distributed training")
    parser.add_argument("--env-id", type=str, required=False, default="Pong-v4", help="Environment ID")
    parser.add_argument("--timesteps", type=int, required=False, default=1000000, help="Total timesteps for training")
    parser.add_argument("--batch-size", type=int, required=False, default=32, help="Training batch size")
    parser.add_argument("--buffer-size", type=int, required=False, default=200000, help="Replay buffer size")
    parser.add_argument("--gamma", type=float, required=False, default=0.99, help="Discount factor")
    parser.add_argument("--target-update-freq", required=False, type=int, default=1000,
                        help="Target network update frequency")
    parser.add_argument("--tau", type=float, required=False, default=1.0,
                        help="Target network update rate (1.0 = hard update)")
    parser.add_argument("--checkpoint-freq", required=False, type=int, default=100000,
                        help="Checkpoint saving frequency")
    parser.add_argument("--eval-freq", required=False, type=int, default=100000, help="Evaluation frequency")

    args = parser.parse_args()

    # Create example observation and action spaces for testing
    # These would match the spaces in the environments created by actors
    observation_space = Box(low=0, high=255, shape=(4, 84, 84), dtype=np.uint8)
    action_space = Discrete(6)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Start trainer
    trainer_loop(
        observation_space=observation_space,
        action_space=action_space,
        env_id=args.env_id,
        total_timesteps=args.timesteps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        device=device,
        target_network_frequency=args.target_update_freq,
        tau=args.tau,
        checkpoint_frequency=args.checkpoint_freq,
        eval_frequency=args.eval_freq,
        buffer_size=args.buffer_size
    )