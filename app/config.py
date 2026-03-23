# Reward and penalty weights for the shift scheduling RL environment.

# ---------------------------------------------------------------------------
# Transition penalties (applied per action in _transition_penalty)
# ---------------------------------------------------------------------------

# Penalty for assigning a shift that differs from the worker's default shift.
PENALTY_NON_DEFAULT_SHIFT: float = 0.2

# Penalty for an illegal/ergonomically poor shift transition:
#   Night -> Day, Night -> Evening, Evening -> Day, Day/Evening -> Night
PENALTY_BAD_TRANSITION: float = 1.0

# Penalty for working 6 or more consecutive days without a day off.
PENALTY_CONSECUTIVE_WORK: float = 1.0

# ---------------------------------------------------------------------------
# Demand gap penalty (applied at episode end in _demand_gap_penalty)
# ---------------------------------------------------------------------------

# Scale factor for each unit of unmet shift demand: weight = DEMAND_GAP_SCALE * num_workers
DEMAND_GAP_SCALE: float = 0.01

# ---------------------------------------------------------------------------
# Per-row scheduling quality penalties (applied in _row_level_penalty)
# ---------------------------------------------------------------------------

# Penalty when the worker has fewer than 2 blocks of consecutive days off.
PENALTY_FEW_CONSECUTIVE_OFF_BLOCKS: float = 0.1

# Penalty when the worker's total days off is below the minimum threshold.
PENALTY_INSUFFICIENT_TOTAL_OFF: float = 0.1

# Minimum number of days off required to avoid PENALTY_INSUFFICIENT_TOTAL_OFF.
MIN_TOTAL_OFF_DAYS: int = 9

# Penalty when the worker has fewer weekend days off than the minimum threshold.
PENALTY_INSUFFICIENT_WEEKEND_OFF: float = 0.1

# Minimum number of weekend days off required to avoid PENALTY_INSUFFICIENT_WEEKEND_OFF.
MIN_WEEKEND_OFF_DAYS: int = 4

# Penalty for having any isolated single-day leave (surrounded by working shifts).
PENALTY_SINGLE_DAY_LEAVE: float = 0.1

# ---------------------------------------------------------------------------
# Step-level rewards / penalties (applied in step)
# ---------------------------------------------------------------------------

# Penalty for taking an action that was masked as invalid.
PENALTY_INVALID_ACTION: float = 1.0

# Reward for assigning a working shift that satisfies current demand.
REWARD_DEMAND_MET: float = 0.05

# Reward for assigning a day off immediately after a previous day off (O -> O).
REWARD_CONSECUTIVE_DAY_OFF: float = 0.02

# Reward for assigning a day off on a weekend day.
REWARD_WEEKEND_DAY_OFF: float = 0.02

# Penalty for assigning a working shift that exceeds current demand.
PENALTY_DEMAND_EXCEEDED: float = 0.05
