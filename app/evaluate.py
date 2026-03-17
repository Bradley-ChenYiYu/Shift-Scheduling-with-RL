"""Standalone schedule loss evaluation script.

Usage examples
--------------
# Evaluate with default paths:
    python app/evaluate.py

# Provide explicit paths:
    python app/evaluate.py \
        --schedule out_Engineer_List.csv \
        --demand app/data/Shift_Demand.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from app.loss_scoring import ScheduleLossEvaluator, print_loss_report
except ModuleNotFoundError:
    from loss_scoring import ScheduleLossEvaluator, print_loss_report


def parse_args() -> argparse.Namespace:
    default_data_dir = Path(__file__).parent / "data"
    parser = argparse.ArgumentParser(
        description="Evaluate a generated shift schedule against loss constraints."
    )
    parser.add_argument(
        "--schedule",
        type=str,
        default="out_Engineer_List.csv",
        help="Path to the schedule CSV produced by the model (default: out_Engineer_List.csv).",
    )
    parser.add_argument(
        "--demand",
        type=str,
        default=str(default_data_dir / "Shift_Demand.csv"),
        help="Path to the shift demand CSV (default: app/data/Shift_Demand.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluator = ScheduleLossEvaluator()
    print(f"Schedule : {args.schedule}")
    print(f"Demand   : {args.demand}")
    print_loss_report(
        schedule_csv_path=args.schedule,
        demand_csv_path=args.demand,
        evaluator=evaluator,
    )


if __name__ == "__main__":
    main()
