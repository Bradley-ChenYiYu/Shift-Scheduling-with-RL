# Shift-Scheduling-with-RL  

## Usage

This workspace uses Docker Compose to run a persistent Python development container.

### Start container

```bash
docker compose -f docker/compose.yaml up -d
```

### Rebuid image and start container

```bash
docker compose -f docker/compose.yaml up -d --build
```

### Stop container

```bash
docker compose -f docker/compose.yaml down
```

### Optional: open a shell in the running container

```bash
docker compose -f docker/compose.yaml exec python-app bash
```

### Optional: run the sample app

```bash
python app/main.py
```

### Optional: compile check

```bash
python -m compileall app/main.py
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train and run masked PPO scheduler

```bash
python app/main.py
```

This will:
- Load `app/data/Engineer_List.csv` and `app/data/Shift_Demand.csv`
- Train a Masked PPO model for scheduling
- Save the model as `ppo_mask.zip`
- Run one inference rollout and print the final schedule

## RL Model Design  

### Observation  

1. The row and column of the cell to predict  
    - Let the model learn to look at the worker's history [Days]  
    - Learn to satisfy the required number of worker [workers]  
2. The amount of required worker [D, E, N]  
3. The amount of worker assinged in each time [D, E, N]  

### Reward (Penalty)  

- Weight = 1.0: Working 6 consecutive days.  
- Weight = 1.0: A Night shift cannot be followed by a Morning or Afternoon shift the next day.  
- Weight = 1.0: An Afternoon shift cannot be followed by a Morning shift the next day.  
- Weight = 1.0: A Morning or Afternoon shift cannot be followed by a Night shift the next day.  
- Weight = 0.2: Violation of default shift patterns.  
- Weight = 0.1: Fewer than 2 instances of consecutive days off per person per month.  
- Weight = 0.1: Fewer than 9 total days off per person per month.  
- Weight = 0.1: Fewer than 4 weekend days off per month.  
- Weight = 0.1: Single-day leave (non-consecutive days off).  

### Separate Loss Scoring

Loss scoring is implemented outside the environment class in `app/loss_scoring.py`:

- `calculate_loss_breakdown(schedule: np.ndarray, default_shifts: np.ndarray, is_weekend: np.ndarray) -> dict[str, float]`
- `calculate_loss_score(schedule: np.ndarray, default_shifts: np.ndarray, is_weekend: np.ndarray) -> float`

This is independent from the reward calculation in `ShiftSchedulingEnv`.

### Action  

- Assign day shift (D)  
- Assign evening shift (E)  
- Assign night shift (N)  

### Model Architecture  

- [Masked PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)  
    - Mask the action if the required man power is met.  

## Input Tables  

Shift Classes: O(Off), D(Day shift), E(Envening Shift), N(Night Shift)

### Engineer_List.csv  

| Person | Default Shift | Date 1 | Date 2 | Date 3
|---|---|---|---|---|
| engineer_1 | D | E | O |   |
| engineer_2 | E | E |   |   |
| engineer_3 | N |   |   |   |

### Shift_Demand.csv

| Date | if_weekend | D | E | N
|---|---|---|---|---|
| Date 1 | Y | 4 | 3 | 2  |
| Date 2 | Y | 4 | 3 | 2  |
| Date 3 |   | 5 | 3 | 2  |
