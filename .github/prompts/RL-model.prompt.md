---
name: RL-model
description: When generating the RL model python library, consider the following design for the model architecture, reward function, and action space. The model should be able to learn from the observations and optimize the scheduling of workers based on the defined rewards and penalties. Use the Stable Baselines3 library for implementing the PPO algorithm with masking to ensure that the model adheres to the constraints of the scheduling problem.
---
## RL Model Design  

In each episode, the model will predict the shift assignment for a specific cell (i, j) in the schedule table. The model will learn to optimize the scheduling of workers based on the defined rewards and penalties. There would be some predefined shifts for some cells based on the Engineer_List.csv, and the model should not change those predefined shifts (jump to the next cell). The model should also learn to satisfy the required number of workers for each shift type (D, E, N) for each date based on the Shift_Demand.csv. Reward are passed after each action. If all the cells in the schedule table are filled, the episode will end.  

### Observation  

If thw model is now predicting the cell at row i and column j, the observation should include:  

1. The i-th row  
2. The j-th column  
3. The amount of required worker [D, E, N] for the j-th column (date)  
4. The amount of worker already assigned for each shift type (D, E, N) for the j-th column (date)   

### Reward (Penalty)  

- Penalty = 1.0: Working 6 consecutive days.  
- Penalty = 1.0: A Night shift cannot be followed by a Morning or Afternoon shift the next day.  
- Penalty = 1.0: An Afternoon shift cannot be followed by a Morning shift the next day.  
- Penalty = 1.0: A Morning or Afternoon shift cannot be followed by a Night shift the next day.  
- Penalty = 0.2: Violation of default shift.  
- Penalty = 0.1: Fewer than 2 instances of consecutive days off per person per month.  
- Penalty = 0.1: Fewer than 9 total days off per person per month.  
- Penalty = 0.1: Fewer than 4 weekend days off per month.  
- Penalty = 0.1: Single-day leave (non-consecutive days off).  

### Action  

- Assign day shift (D) as value 0  
- Assign evening shift (E) as value 1  
- Assign night shift (N) as value 2  
- Assign off (O) as value 3  

### Model Architecture  

- [Masked PPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html)  
    - Mask:  
      - Action if the required man power is met.  
      - If cell (i, j-1) is N, mask action D and E for cell (i, j).  
      - If cell (i, j-1) is E, mask action D for cell (i, j).  
      - If cell (i, j-1) is D or E, mask action N for cell (i, j).  

## Input Tables  

Shift Classes: O(Off), D(Day shift), E(Envening Shift), N(Night Shift)  

### Engineer_List.csv  

The model should read the Engineer_List.csv to get the default shifts for each engineer and the jump the predefined shifts for some cells.  

| Person | Default Shift | Date 1 | Date 2 | Date 3
|---|---|---|---|---|
| engineer_1 | D | E | O |   |
| engineer_2 | E | E |   |   |
| engineer_3 | N |   |   |   |

### Shift_Demand.csv  

The model should read the Shift_Demand.csv to get the required number of workers for each shift type (D, E, N) for each date and learn to satisfy the demand.  

| Date | if_weekend | D | E | N
|---|---|---|---|---|
| Date 1 | Y | 4 | 3 | 2  |
| Date 2 | Y | 4 | 3 | 2  |
| Date 3 |   | 5 | 3 | 2  |

## Environment Interface  

Using the Stable Baselines3 library for implementing the PPO algorithm with masking, please use the following example code to create the custom environment for the scheduling problem from [link](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).  

Please modify the requirements.txt to include the necessary libraries for the custom environment and the PPO algorithm with masking.

Define and train the model:  

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback


env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
model.learn(5_000)

evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

model.save("ppo_mask")
del model # remove to demonstrate saving and loading

model = MaskablePPO.load("ppo_mask")

obs, _ = env.reset()
while True:
    # Retrieve current action mask
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)
```

Define the masking function for the PPO model:  

```python
from sb3_contrib import MaskablePPO
from sb3_contrib.common.envs import InvalidActionEnvDiscrete
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback


env = InvalidActionEnvDiscrete(dim=80, n_invalid_actions=60)
model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
model.learn(5_000)

evaluate_policy(model, env, n_eval_episodes=20, reward_threshold=90, warn=False)

model.save("ppo_mask")
del model # remove to demonstrate saving and loading

model = MaskablePPO.load("ppo_mask")

obs, _ = env.reset()
while True:
    # Retrieve current action mask
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)
```