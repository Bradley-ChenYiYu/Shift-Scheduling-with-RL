from __future__ import annotations

import argparse
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


def parse_bool(value: str) -> bool:
	value_normalized = str(value).strip().lower()
	if value_normalized in {"1", "true", "t", "yes", "y"}:
		return True
	if value_normalized in {"0", "false", "f", "no", "n"}:
		return False
	raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}. Use true/false.")


def parse_args() -> argparse.Namespace:
	default_data_dir = Path(__file__).parent / "data"
	parser = argparse.ArgumentParser(description="Train or evaluate the shift scheduling model.")
	parser.add_argument(
		"--run-model",
		type=parse_bool,
		default=True,
		help="Whether to run model training + rollout before evaluation (default: true). Use --run-model=false to evaluate an existing schedule CSV.",
	)
	parser.add_argument(
		"--schedule",
		type=str,
		default="out_Engineer_List.csv",
		help="Path to schedule CSV to write (when run-model=true) or read (when run-model=false).",
	)
	parser.add_argument(
		"--demand",
		type=str,
		default=str(default_data_dir / "Shift_Demand.csv"),
		help="Path to shift demand CSV for loss evaluation.",
	)
	parser.add_argument(
		"--timesteps",
		type=int,
		default=500_000,
		help="Training timesteps when run-model=true (default: 500000).",
	)
	parser.add_argument(
		"--model-path",
		type=str,
		default="ppo_mask",
		help="Model save prefix for training output (default: ppo_mask).",
	)
	parser.add_argument(
		"--inference-model",
		type=str,
		default="best_model",
		help="Model path used for rollout inference (default: best_model).",
	)
	return parser.parse_args()


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


def rollout_once(
	model: MaskablePPO,
	engineer_df: pd.DataFrame,
	demand_df: pd.DataFrame,
	schedule_output_path: str = "out_Engineer_List.csv",
) -> str:
	env = ShiftSchedulingEnv(engineer_df=engineer_df, demand_df=demand_df)
	obs, _ = env.reset()

	while True:
		action_masks = get_action_masks(env)
		action, _ = model.predict(obs, action_masks=action_masks, deterministic=True)
		obs, reward, terminated, truncated, _ = env.step(int(action))
		if terminated or truncated:
			break

	print("Final schedule:")
	print(env.render(output_path=schedule_output_path))
	return schedule_output_path


def main() -> None:
	args = parse_args()
	data_dir = Path(__file__).parent / "data"
	if args.run_model:
		engineer_df, demand_df = load_input_tables(data_dir)
		train_model(
			engineer_df=engineer_df,
			demand_df=demand_df,
			timesteps=args.timesteps,
			model_path=args.model_path,
		)
		model = MaskablePPO.load(args.inference_model)
		schedule_path = rollout_once(
			model,
			engineer_df=engineer_df,
			demand_df=demand_df,
			schedule_output_path=args.schedule,
		)
	else:
		schedule_path = args.schedule

	print_loss_report(
		schedule_csv_path=schedule_path,
		demand_csv_path=args.demand,
	)


if __name__ == "__main__":
	main()
