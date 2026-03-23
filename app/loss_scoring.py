from __future__ import annotations

import numpy as np
import pandas as pd


SHIFT_TO_ID = {"D": 0, "E": 1, "N": 2, "O": 3}
WORKING_SHIFT_IDS = {SHIFT_TO_ID["D"], SHIFT_TO_ID["E"], SHIFT_TO_ID["N"]}

PERSON_COLUMN_ALIASES = ["Person", "人員"]
DEFAULT_SHIFT_COLUMN_ALIASES = ["Default Shift", "班別群組(*第一碼為群組代碼第二碼之後為可backup群組)"]
DEMAND_DATE_COLUMN_ALIASES = ["Date"]
WEEKEND_COLUMN_ALIASES = ["if_weekend", "IfWeekend"]


class ScheduleLossEvaluator:
    def __init__(
        self,
        weight_working_6_consecutive_days: float = 1.0,
        weight_night_to_morning_or_afternoon: float = 1.0,
        weight_afternoon_to_morning: float = 1.0,
        weight_morning_or_afternoon_to_night: float = 1.0,
        weight_default_shift_pattern_violation: float = 0.2,
        weight_fewer_than_2_consecutive_days_off_instances: float = 0.1,
        weight_fewer_than_9_total_days_off: float = 0.1,
        weight_fewer_than_4_weekend_days_off: float = 0.1,
        weight_single_day_leave: float = 0.1,
        min_total_off_days: int = 9,
        min_weekend_off_days: int = 4,
        min_consecutive_off_instances: int = 2,
    ):
        self.weight_working_6_consecutive_days = weight_working_6_consecutive_days
        self.weight_night_to_morning_or_afternoon = weight_night_to_morning_or_afternoon
        self.weight_afternoon_to_morning = weight_afternoon_to_morning
        self.weight_morning_or_afternoon_to_night = weight_morning_or_afternoon_to_night
        self.weight_default_shift_pattern_violation = weight_default_shift_pattern_violation
        self.weight_fewer_than_2_consecutive_days_off_instances = weight_fewer_than_2_consecutive_days_off_instances
        self.weight_fewer_than_9_total_days_off = weight_fewer_than_9_total_days_off
        self.weight_fewer_than_4_weekend_days_off = weight_fewer_than_4_weekend_days_off
        self.weight_single_day_leave = weight_single_day_leave

        self.min_total_off_days = min_total_off_days
        self.min_weekend_off_days = min_weekend_off_days
        self.min_consecutive_off_instances = min_consecutive_off_instances

    @staticmethod
    def _resolve_column(frame: pd.DataFrame, aliases: list[str], column_role: str) -> str:
        for alias in aliases:
            if alias in frame.columns:
                return alias
        raise ValueError(
            f"Missing {column_role} column. Expected one of: {aliases}. Actual columns: {list(frame.columns)}"
        )

    def _count_consecutive_work_violations(self, row: np.ndarray) -> int:
        violations = 0
        consecutive_work = 0
        for shift_id in row:
            if shift_id in WORKING_SHIFT_IDS:
                consecutive_work += 1
                if consecutive_work >= 6:
                    violations += 1
            else:
                consecutive_work = 0
        return int(violations)


    def _row_offday_metrics(self, row: np.ndarray, is_weekend: np.ndarray) -> tuple[int, int, bool, int]:
        off_mask = row == SHIFT_TO_ID["O"]
        total_off = int(np.sum(off_mask))

        consecutive_off_instances = 0
        has_single_day_leave = False
        index = 0
        num_days = row.shape[0]
        while index < num_days:
            if not off_mask[index]:
                index += 1
                continue

            end = index
            while end < num_days and off_mask[end]:
                end += 1

            run_len = end - index
            if run_len >= 2:
                consecutive_off_instances += 1
            if run_len == 1:
                has_single_day_leave = True

            index = end

        weekend_off = int(np.sum(off_mask & is_weekend))
        return total_off, consecutive_off_instances, has_single_day_leave, weekend_off

    def _shift_labels_to_ids(self, frame: pd.DataFrame, date_columns: list[str]) -> np.ndarray:
        schedule = np.full((len(frame), len(date_columns)), SHIFT_TO_ID["O"], dtype=np.int32)
        for worker_idx, row in frame.iterrows():
            for day_idx, date_col in enumerate(date_columns):
                value = row[date_col]
                if pd.isna(value) or str(value).strip() == "":
                    shift_label = "O"
                else:
                    shift_label = str(value).strip().upper()
                if shift_label not in SHIFT_TO_ID:
                    raise ValueError(f"Invalid shift '{shift_label}' in column {date_col}")
                schedule[worker_idx, day_idx] = SHIFT_TO_ID[shift_label]
        return schedule

    def _default_shift_labels_to_ids(self, default_shifts: pd.Series) -> np.ndarray:
        values: list[int] = []
        for value in default_shifts:
            shift_label = str(value).strip().upper()
            if shift_label not in SHIFT_TO_ID:
                shift_label = shift_label[:1]
            if shift_label not in SHIFT_TO_ID:
                raise ValueError(f"Invalid default shift: {value}")
            values.append(SHIFT_TO_ID[shift_label])
        return np.array(values, dtype=np.int32)


    def evaluate_from_arrays(
        self,
        schedule: np.ndarray,
        default_shifts: np.ndarray,
        is_weekend: np.ndarray,
    ) -> dict[str, float]:
        if schedule.ndim != 2:
            raise ValueError(f"Expected 2D schedule array, got ndim={schedule.ndim}")
        if default_shifts.shape[0] != schedule.shape[0]:
            raise ValueError(
                f"default_shifts length ({default_shifts.shape[0]}) must match workers ({schedule.shape[0]})"
            )
        if is_weekend.shape[0] != schedule.shape[1]:
            raise ValueError(
                f"is_weekend length ({is_weekend.shape[0]}) must match days ({schedule.shape[1]})"
            )

        components = {
            "working_6_consecutive_days": 0.0,
            "night_to_morning_or_afternoon": 0.0,
            "afternoon_to_morning": 0.0,
            "morning_or_afternoon_to_night": 0.0,
            "default_shift_pattern_violation": 0.0,
            "fewer_than_2_consecutive_days_off_instances": 0.0,
            "fewer_than_9_total_days_off": 0.0,
            "fewer_than_4_weekend_days_off": 0.0,
            "single_day_leave": 0.0,
        }

        num_workers, num_days = schedule.shape
        for worker_idx in range(num_workers):
            row = schedule[worker_idx, :]
            default_shift = int(default_shifts[worker_idx])

            consecutive_work_violations = self._count_consecutive_work_violations(row)
            components["working_6_consecutive_days"] += (
                self.weight_working_6_consecutive_days * consecutive_work_violations
            )

            for day_idx in range(1, num_days):
                previous_shift = row[day_idx - 1]
                current_shift = row[day_idx]

                if previous_shift == SHIFT_TO_ID["N"] and current_shift in (SHIFT_TO_ID["D"], SHIFT_TO_ID["E"]):
                    components["night_to_morning_or_afternoon"] += self.weight_night_to_morning_or_afternoon
                if previous_shift == SHIFT_TO_ID["E"] and current_shift == SHIFT_TO_ID["D"]:
                    components["afternoon_to_morning"] += self.weight_afternoon_to_morning
                if previous_shift in (SHIFT_TO_ID["D"], SHIFT_TO_ID["E"]) and current_shift == SHIFT_TO_ID["N"]:
                    components["morning_or_afternoon_to_night"] += self.weight_morning_or_afternoon_to_night

            non_default_assignments = np.sum((row != SHIFT_TO_ID["O"]) & (row != default_shift))
            components["default_shift_pattern_violation"] += (
                self.weight_default_shift_pattern_violation * int(non_default_assignments)
            )

            total_off, consecutive_off_instances, has_single_day_leave, weekend_off = self._row_offday_metrics(
                row, is_weekend
            )
            if consecutive_off_instances < self.min_consecutive_off_instances:
                components["fewer_than_2_consecutive_days_off_instances"] += (
                    self.weight_fewer_than_2_consecutive_days_off_instances
                )
            if total_off < self.min_total_off_days:
                components["fewer_than_9_total_days_off"] += self.weight_fewer_than_9_total_days_off
            if weekend_off < self.min_weekend_off_days:
                components["fewer_than_4_weekend_days_off"] += self.weight_fewer_than_4_weekend_days_off
            if has_single_day_leave:
                components["single_day_leave"] += self.weight_single_day_leave

        components["total_loss"] = float(sum(components.values()))
        return components

    def evaluate_score_from_arrays(
        self,
        schedule: np.ndarray,
        default_shifts: np.ndarray,
        is_weekend: np.ndarray,
    ) -> float:
        return float(
            self.evaluate_from_arrays(
                schedule=schedule,
                default_shifts=default_shifts,
                is_weekend=is_weekend,
            )["total_loss"]
        )

    def evaluate_from_csv(self, schedule_csv_path: str, demand_csv_path: str) -> dict[str, float]:
        schedule_df = pd.read_csv(schedule_csv_path)
        demand_df = pd.read_csv(demand_csv_path)

        person_column = self._resolve_column(schedule_df, PERSON_COLUMN_ALIASES, "schedule person")
        default_shift_column = self._resolve_column(
            schedule_df,
            DEFAULT_SHIFT_COLUMN_ALIASES,
            "schedule default shift",
        )
        demand_date_column = self._resolve_column(demand_df, DEMAND_DATE_COLUMN_ALIASES, "demand date")
        weekend_column = self._resolve_column(demand_df, WEEKEND_COLUMN_ALIASES, "demand weekend")

        date_columns = [
            column
            for column in schedule_df.columns
            if column not in [person_column, default_shift_column]
        ]
        if not date_columns:
            raise ValueError("Schedule CSV must contain date columns.")

        demand_dates = list(demand_df[demand_date_column].astype(str))
        if [str(column) for column in date_columns] != demand_dates:
            raise ValueError(
                "Date columns mismatch between schedule CSV and demand CSV. "
                f"schedule dates={[str(column) for column in date_columns]}, demand dates={demand_dates}"
            )

        schedule = self._shift_labels_to_ids(schedule_df, date_columns)
        default_shifts = self._default_shift_labels_to_ids(schedule_df[default_shift_column])
        is_weekend = (
            demand_df[weekend_column].fillna("").astype(str).str.strip().str.upper() == "Y"
        ).to_numpy(dtype=bool)

        return self.evaluate_from_arrays(
            schedule=schedule,
            default_shifts=default_shifts,
            is_weekend=is_weekend,
        )

    def evaluate_score_from_csv(self, schedule_csv_path: str, demand_csv_path: str) -> float:
        return float(self.evaluate_from_csv(schedule_csv_path=schedule_csv_path, demand_csv_path=demand_csv_path)["total_loss"])


def print_loss_report(
    schedule_csv_path: str,
    demand_csv_path: str,
    evaluator: ScheduleLossEvaluator | None = None,
) -> dict[str, float]:
    if evaluator is None:
        evaluator = ScheduleLossEvaluator()
    breakdown = evaluator.evaluate_from_csv(
        schedule_csv_path=schedule_csv_path,
        demand_csv_path=demand_csv_path,
    )
    print("\nLoss score (separate from reward):")
    for key, value in breakdown.items():
        print(f"  {key}: {value:.3f}")
    return breakdown


def calculate_loss_breakdown(
    schedule: np.ndarray,
    default_shifts: np.ndarray,
    is_weekend: np.ndarray,
) -> dict[str, float]:
    evaluator = ScheduleLossEvaluator()
    return evaluator.evaluate_from_arrays(
        schedule=schedule,
        default_shifts=default_shifts,
        is_weekend=is_weekend,
    )


def calculate_loss_score(schedule: np.ndarray, default_shifts: np.ndarray, is_weekend: np.ndarray) -> float:
    evaluator = ScheduleLossEvaluator()
    return evaluator.evaluate_score_from_arrays(
        schedule=schedule,
        default_shifts=default_shifts,
        is_weekend=is_weekend,
    )
