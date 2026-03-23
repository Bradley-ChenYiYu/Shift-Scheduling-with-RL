from __future__ import annotations

from pathlib import Path

import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.monitor import Monitor

try:
	from app.rl_env import ShiftSchedulingEnv
	from app.loss_scoring import print_loss_report
except ModuleNotFoundError:
	from rl_env import ShiftSchedulingEnv
	from loss_scoring import print_loss_report


def load_input_tables(base_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
	engineer_path = base_dir / "Engineer_List.csv"
	demand_path = base_dir / "Shift_Demand.csv"

	engineer_df = pd.read_csv(engineer_path)
	demand_df = pd.read_csv(demand_path)
	return engineer_df, demand_df


def train_model(
	engineer_df: pd.DataFrame,
	demand_df: pd.DataFrame,
	timesteps: int = 5_000,
	model_path: str = "ppo_mask",
) -> MaskablePPO:
	env = Monitor(ShiftSchedulingEnv(engineer_df=engineer_df, demand_df=demand_df))
	eval_env = Monitor(ShiftSchedulingEnv(engineer_df=engineer_df, demand_df=demand_df))

	eval_callback = MaskableEvalCallback(
		eval_env,
		best_model_save_path=".",
		log_path=".",
		eval_freq=1_000,
		n_eval_episodes=5,
		deterministic=True,
	)

	model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1, tensorboard_log = "./ppo_mask_tensorboard/")
	model.learn(total_timesteps=timesteps, callback=eval_callback)

	mean_reward, std_reward = evaluate_policy(
		model,
		eval_env,
		n_eval_episodes=20,
		warn=False,
	)
	print(f"Evaluation reward: mean={mean_reward:.3f}, std={std_reward:.3f}")

	model.save(model_path)
	return model


def rollout_once(model: MaskablePPO, engineer_df: pd.DataFrame, demand_df: pd.DataFrame) -> None:
	env = ShiftSchedulingEnv(engineer_df=engineer_df, demand_df=demand_df)
	obs, _ = env.reset()

	while True:
		action_masks = get_action_masks(env)
		action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
		obs, reward, terminated, truncated, _ = env.step(int(action))
		if terminated or truncated:
			break

	print("Final schedule:")
	schedule_output_path = "out_Engineer_List.csv"
	print(env.render(output_path=schedule_output_path))
	print_loss_report(
		schedule_csv_path=schedule_output_path,
		demand_csv_path=str(Path(__file__).parent / "data" / "Shift_Demand.csv"),
	)


def main() -> None:
	data_dir = Path(__file__).parent / "data"
	engineer_df, demand_df = load_input_tables(data_dir)

	train_model(engineer_df=engineer_df, demand_df=demand_df, timesteps=500_000, model_path="ppo_mask") # 500_000
	model = MaskablePPO.load("best_model")

	rollout_once(model, engineer_df=engineer_df, demand_df=demand_df)


if __name__ == "__main__":
	main()
