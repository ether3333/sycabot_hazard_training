import argparse

from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from sycabot_env import SycaBotEnv

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--num-robots", type=int, default=2)
    parser.add_argument("--num-tasks", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=1000)

    parser.add_argument("--fire-spread-prob", type=float, default=0.020)
    parser.add_argument("--fire-kill-prob", type=float, default=0.2)

    parser.add_argument("--pickup-reward", type=float, default=1000.0)
    parser.add_argument("--delivery-reward", type=float, default=1000.0)

    parser.add_argument("--task-progress-weight", type=float, default=20.0)
    parser.add_argument("--exit-progress-weight", type=float, default=20.0)

    parser.add_argument("--total-timesteps", type=int, default=20000)

    return parser.parse_args()


class RewardComponentTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.progress_vals = []
        self.pickup_vals = []
        self.delivery_vals = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "reward_progress" in info:
                self.progress_vals.append(float(info["reward_progress"]))
            if "reward_pickup" in info:
                self.pickup_vals.append(float(info["reward_pickup"]))
            if "reward_delivery" in info:
                self.delivery_vals.append(float(info["reward_delivery"]))
        return True

    def _on_rollout_end(self) -> None:
        if self.progress_vals:
            self.logger.record("reward_components/progress", sum(self.progress_vals) / len(self.progress_vals))
            self.progress_vals.clear()
        if self.pickup_vals:
            self.logger.record("reward_components/pickup", sum(self.pickup_vals) / len(self.pickup_vals))
            self.pickup_vals.clear()
        if self.delivery_vals:
            self.logger.record("reward_components/delivery", sum(self.delivery_vals) / len(self.delivery_vals))
            self.delivery_vals.clear()


CONTINUE_FROM_PREVIOUS = True # allows warm start (policy from previous training)

args = parse_args()

if args.run_name is None:
    args.run_name = datetime.now().strftime("ppo_sycabot_%Y%m%d_%H%M%S")

env = SycaBotEnv(
    render_mode=None, #no needto visualize during training
    num_robots=1,
    num_tasks=1,
    fire_spread_prob=0.02,
    fire_kill_prob=0.2,
    fire_cell_size=0.08,
)

tensorboard_log = "./ppo_sycabot_tensorboard/"
device = "cuda" if hasattr(env, "device") and env.device == "cuda" else "cpu"
base_model_path = Path("ppo_sycabot.zip") # rename to start from a specific file.

if CONTINUE_FROM_PREVIOUS and base_model_path.exists():
    model = PPO.load(
        str(base_model_path),
        env=env,
        device=device,
        tensorboard_log=tensorboard_log,
    )
else:
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        device=device,
    )

component_callback = RewardComponentTensorboardCallback()
model.learn(
    total_timesteps=args.total_timesteps,
    progress_bar=True,
    callback=component_callback,
    reset_num_timesteps=not CONTINUE_FROM_PREVIOUS,
    tb_log_name=args.run_name,
)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

model.save(model_dir / f"ppo_sycabot_{args.run_name}")

# tensorboard --logdir ./ppo_sycabot_tensorboard/
