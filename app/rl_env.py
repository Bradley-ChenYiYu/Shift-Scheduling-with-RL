from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
from config import (
    DEMAND_GAP_SCALE,
    MIN_TOTAL_OFF_DAYS,
    MIN_WEEKEND_OFF_DAYS,
    PENALTY_BAD_TRANSITION,
    PENALTY_CONSECUTIVE_WORK,
    PENALTY_DEMAND_EXCEEDED,
    PENALTY_FEW_CONSECUTIVE_OFF_BLOCKS,
    PENALTY_INSUFFICIENT_TOTAL_OFF,
    PENALTY_INSUFFICIENT_WEEKEND_OFF,
    PENALTY_INVALID_ACTION,
    PENALTY_NON_DEFAULT_SHIFT,
    PENALTY_SINGLE_DAY_LEAVE,
    REWARD_DEMAND_MET,
)
import numpy as np
import pandas as pd
import warnings


SHIFT_TO_ID = {"D": 0, "E": 1, "N": 2, "O": 3}
ID_TO_SHIFT = {value: key for key, value in SHIFT_TO_ID.items()}
WORKING_SHIFT_IDS = {SHIFT_TO_ID["D"], SHIFT_TO_ID["E"], SHIFT_TO_ID["N"]}


@dataclass
class EngineerConfig:
    name: str
    default_shift: int


class ShiftSchedulingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, engineer_df: pd.DataFrame, demand_df: pd.DataFrame):
        super().__init__()

        self.engineer_df = engineer_df.copy()
        self.demand_df = demand_df.copy()

        self.date_columns = [column for column in self.engineer_df.columns if column not in ["Person", "Default Shift"]]
        if not self.date_columns:
            raise ValueError("Engineer_List.csv must contain date columns.")

        self.num_workers = len(self.engineer_df)
        self.num_days = len(self.date_columns)

        self._validate_input_tables()

        self.engineers = self._build_engineer_config()
        self.demand = self.demand_df[["D", "E", "N"]].to_numpy(dtype=np.int32)
        self.is_weekend = (
            self.demand_df["if_weekend"].fillna("").astype(str).str.strip().str.upper() == "Y"
        ).to_numpy(dtype=bool)

        self.predefined = self._build_predefined_schedule()
        self.locked = self.predefined != -1

        self.action_space = gym.spaces.Discrete(4)
        # Observation:
        # [i-th row], [j-th column], [required_D, required_E, required_N],
        # [assigned_D, assigned_E, assigned_N], [is_weekend(day_1..day_n)]
        obs_size = self.num_days + self.num_workers + 3 + 3 + self.num_days
        # Row and column values are -1 for unassigned, otherwise the shift ID: 0~3.
        self.observation_space = gym.spaces.Box(
            low=-1.0,
            high=3.0,
            shape=(obs_size,),
            dtype=np.float32,
        )

        self.schedule = np.full((self.num_workers, self.num_days), -1, dtype=np.int32)
        self.current_i = 0
        self.current_j = 0
        self.terminated = False
        self.evaluated_rows: set[int] = set()

    def _validate_input_tables(self) -> None:
        expected_engineer = {"Person", "Default Shift"}
        missing_engineer = expected_engineer - set(self.engineer_df.columns)
        if missing_engineer:
            raise ValueError(f"Engineer_List.csv missing columns: {missing_engineer}")

        expected_demand = {"Date", "if_weekend", "D", "E", "N"}
        missing_demand = expected_demand - set(self.demand_df.columns)
        if missing_demand:
            raise ValueError(f"Shift_Demand.csv missing columns: {missing_demand}")

        demand_dates = list(self.demand_df["Date"].astype(str))
        engineer_dates = [str(column) for column in self.date_columns]
        if demand_dates != engineer_dates:
            raise ValueError(
                "Date columns mismatch between Engineer_List.csv and Shift_Demand.csv. "
                f"Engineer dates={engineer_dates}, demand dates={demand_dates}"
            )

    def _build_engineer_config(self) -> list[EngineerConfig]:
        engineers: list[EngineerConfig] = []
        for _, row in self.engineer_df.iterrows():
            default_shift_label = str(row["Default Shift"]).strip().upper()
            if default_shift_label not in SHIFT_TO_ID:
                raise ValueError(f"Invalid default shift: {default_shift_label}")

            engineers.append(
                EngineerConfig(
                    name=str(row["Person"]),
                    default_shift=SHIFT_TO_ID[default_shift_label],
                )
            )
        return engineers

    def _build_predefined_schedule(self) -> np.ndarray:
        predefined = np.full((self.num_workers, self.num_days), -1, dtype=np.int32)
        for worker_idx, row in self.engineer_df.iterrows():
            for day_idx, date_col in enumerate(self.date_columns):
                value = row[date_col]
                if pd.isna(value) or str(value).strip() == "":
                    continue

                shift_label = str(value).strip().upper()
                if shift_label not in SHIFT_TO_ID:
                    raise ValueError(f"Invalid predefined shift '{shift_label}' in column {date_col}")
                predefined[worker_idx, day_idx] = SHIFT_TO_ID[shift_label]
        return predefined

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        self.schedule = np.full((self.num_workers, self.num_days), -1, dtype=np.int32)
        self.schedule[self.locked] = self.predefined[self.locked]
        self.evaluated_rows = set()
        self.terminated = False

        self.current_i, self.current_j = self._find_next_cell(0, -1)
        if self.current_i == -1:
            self.terminated = True

        return self._get_obs(), {}

    def _find_next_cell(self, start_i: int, start_j: int) -> tuple[int, int]:
        if start_j < 0:
            flat_start = 0
        else:
            flat_start = start_j * self.num_workers + (start_i + 1)
        for index in range(flat_start, self.num_workers * self.num_days):
            day_idx = index // self.num_workers
            worker_idx = index % self.num_workers
            if self.schedule[worker_idx, day_idx] == -1 and not self.locked[worker_idx, day_idx]:
                return worker_idx, day_idx
        return -1, -1

    def _get_obs(self) -> np.ndarray:
        if self.terminated:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        row = self.schedule[self.current_i, :].astype(np.float32)
        col = self.schedule[:, self.current_j].astype(np.float32)
        required = self.demand[self.current_j].astype(np.float32)
        assigned = np.array(self._assigned_counts(self.current_j), dtype=np.float32)
        weekend_flags = self.is_weekend.astype(np.float32)

        return np.concatenate([row, col, required, assigned, weekend_flags]).astype(np.float32)

    def _assigned_counts(self, day_idx: int) -> tuple[int, int, int]:
        day_assignments = self.schedule[:, day_idx]
        return (
            int(np.sum(day_assignments == SHIFT_TO_ID["D"])),
            int(np.sum(day_assignments == SHIFT_TO_ID["E"])),
            int(np.sum(day_assignments == SHIFT_TO_ID["N"])),
        )

    def _demand_gap_penalty(self) -> float:
        penalty = 0.0
        for day_idx in range(self.num_days):
            assigned = self._assigned_counts(day_idx)
            demand = self.demand[day_idx]
            for shift_id in WORKING_SHIFT_IDS:
                penalty += DEMAND_GAP_SCALE * self.num_workers * (demand[shift_id] - assigned[shift_id])
        return float(penalty)

    def action_masks(self) -> np.ndarray:
        if self.terminated:
            return np.array([True, True, True, True], dtype=bool)

        masks = np.ones(self.action_space.n, dtype=bool)

        assigned_d, assigned_e, assigned_n = self._assigned_counts(self.current_j)
        demand_d, demand_e, demand_n = self.demand[self.current_j]

        if assigned_d >= demand_d:
            masks[SHIFT_TO_ID["D"]] = False
        if assigned_e >= demand_e:
            masks[SHIFT_TO_ID["E"]] = False
        if assigned_n >= demand_n:
            masks[SHIFT_TO_ID["N"]] = False

        if self.current_j > 0:
            previous_shift = self.schedule[self.current_i, self.current_j - 1]
            if previous_shift == SHIFT_TO_ID["N"]:
                masks[SHIFT_TO_ID["D"]] = False
                masks[SHIFT_TO_ID["E"]] = False
            if previous_shift == SHIFT_TO_ID["E"]:
                masks[SHIFT_TO_ID["D"]] = False
            if previous_shift in (SHIFT_TO_ID["D"], SHIFT_TO_ID["E"]):
                masks[SHIFT_TO_ID["N"]] = False

        if not masks.any():
            masks[SHIFT_TO_ID["O"]] = True

        return masks

    def _transition_penalty(self, worker_idx: int, day_idx: int, shift_id: int) -> float:
        penalty = 0.0
        default_shift = self.engineers[worker_idx].default_shift

        if shift_id != SHIFT_TO_ID["O"] and shift_id != default_shift:
            penalty += PENALTY_NON_DEFAULT_SHIFT

        if day_idx > 0:
            previous_shift = self.schedule[worker_idx, day_idx - 1]

            if previous_shift == SHIFT_TO_ID["N"] and shift_id in (SHIFT_TO_ID["D"], SHIFT_TO_ID["E"]):
                penalty += PENALTY_BAD_TRANSITION
            if previous_shift == SHIFT_TO_ID["E"] and shift_id == SHIFT_TO_ID["D"]:
                penalty += PENALTY_BAD_TRANSITION
            if previous_shift in (SHIFT_TO_ID["D"], SHIFT_TO_ID["E"]) and shift_id == SHIFT_TO_ID["N"]:
                penalty += PENALTY_BAD_TRANSITION

        if shift_id in WORKING_SHIFT_IDS and day_idx >= 5:
            recent = self.schedule[worker_idx, day_idx - 5 : day_idx]
            if np.all(np.isin(recent, list(WORKING_SHIFT_IDS))):
                penalty += PENALTY_CONSECUTIVE_WORK

        return penalty

    def _row_level_penalty(self, worker_idx: int) -> float:
        row = self.schedule[worker_idx, :]
        penalty = 0.0

        off_mask = row == SHIFT_TO_ID["O"]
        total_off = int(np.sum(off_mask))

        consecutive_off_instances = 0
        has_single_day_leave = False
        index = 0
        while index < self.num_days:
            if not off_mask[index]:
                index += 1
                continue

            end = index
            while end < self.num_days and off_mask[end]:
                end += 1

            run_len = end - index
            if run_len >= 2:
                consecutive_off_instances += 1
            if run_len == 1:
                has_single_day_leave = True

            index = end

        weekend_off = int(np.sum(off_mask & self.is_weekend))

        if consecutive_off_instances < 2:
            penalty += PENALTY_FEW_CONSECUTIVE_OFF_BLOCKS
        if total_off < MIN_TOTAL_OFF_DAYS:
            penalty += PENALTY_INSUFFICIENT_TOTAL_OFF
        if weekend_off < MIN_WEEKEND_OFF_DAYS:
            penalty += PENALTY_INSUFFICIENT_WEEKEND_OFF
        if has_single_day_leave:
            penalty += PENALTY_SINGLE_DAY_LEAVE

        return penalty

    def step(self, action: int):
        if self.terminated:
            return self._get_obs(), 0.0, True, False, {}

        worker_idx, day_idx = self.current_i, self.current_j
        reward = 0.0

        masks = self.action_masks()
        # This should not happen if the action mask is applied correctly, send a warning and assign a penalty if it does.
        if not masks[action]:
            warnings.warn(
                f"Selected masked action {action} at worker {worker_idx}, day {day_idx}",
                stacklevel=2,
            )
            reward -= PENALTY_INVALID_ACTION 
            valid_actions = np.where(masks)[0]
            if len(valid_actions) > 0:
                action = int(valid_actions[0])
            else:
                action = SHIFT_TO_ID["O"]

        reward -= self._transition_penalty(worker_idx, day_idx, int(action))
        self.schedule[worker_idx, day_idx] = int(action)

        assigned = self._assigned_counts(day_idx)
        demand = self.demand[day_idx]
        if action in WORKING_SHIFT_IDS and assigned[action] <= demand[action]:
            reward += REWARD_DEMAND_MET
        elif action in WORKING_SHIFT_IDS and assigned[action] > demand[action]:
            reward -= PENALTY_DEMAND_EXCEEDED

        next_i, next_j = self._find_next_cell(worker_idx, day_idx)

        if worker_idx not in self.evaluated_rows and np.all(self.schedule[worker_idx, ~self.locked[worker_idx]] != -1):
            reward -= self._row_level_penalty(worker_idx)
            self.evaluated_rows.add(worker_idx)

        if next_i == -1:
            self.terminated = True
            for row_idx in range(self.num_workers):
                if row_idx not in self.evaluated_rows:
                    reward -= self._row_level_penalty(row_idx)
                    self.evaluated_rows.add(row_idx)
            reward -= self._demand_gap_penalty()
            return self._get_obs(), float(reward), True, False, {}

        self.current_i, self.current_j = next_i, next_j
        return self._get_obs(), float(reward), False, False, {}

    def render(self, output_path: str = "out_Engineer_List.csv"):
        schedule_labels = np.vectorize(lambda value: ID_TO_SHIFT.get(int(value), ""))(self.schedule)
        frame = pd.DataFrame(schedule_labels, columns=self.date_columns)
        frame.insert(0, "Person", [engineer.name for engineer in self.engineers])
        frame.insert(1, "Default Shift", self.engineer_df["Default Shift"].astype(str).tolist())
        frame.to_csv(output_path, index=False)
        return frame