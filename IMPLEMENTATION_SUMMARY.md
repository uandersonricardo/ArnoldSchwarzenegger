# Game Features Prediction Implementation Summary

## Changes Applied

This document summarizes the modifications made to implement game features (monster/enemy) prediction in LevDoom, similar to Arnold's auxiliary task approach.

### 1. Network Architecture (`src/levd/network.py`)

**Changes to `Rainbow` class:**
- Added parameters: `n_features` (int) and `hidden_dim` (int, default 512)
- Created auxiliary prediction head when `n_features > 0`:
  ```python
  self.proj_game_features = nn.Sequential(
      nn.Dropout(0.5),
      nn.Linear(self.output_dim, hidden_dim),
      nn.ReLU(),
      nn.Dropout(0.5),
      nn.Linear(hidden_dim, self.n_features),
      nn.Sigmoid()  # Binary predictions for each feature
  )
  ```
- Modified `forward()` to return tuple `(probs, features), state` when features are enabled

### 2. Environment Base Class (`libs/LevDoom/levdoom/envs/base.py`)

**New parameters:**
- `use_labels: bool = False` - Enable ViZDoom labels buffer
- `game_features: List[str] = None` - List of features to track

**New method `_extract_game_features(state)`:**
- Extracts object labels from ViZDoom state
- Classifies objects into categories (enemy, health, weapon, ammo)
- Returns binary vector indicating presence of each feature type
- Handles errors gracefully by returning zeros

**Object Sets Defined:**
- `ENEMY_SET`: All marine types, demons, monsters (25+ enemy types)
- `HEALTH_SET`: Health pickups, armor, powerups
- `WEAPON_SET`: All weapons (pistol to BFG9000)
- `AMMO_SET`: All ammunition types

**Modified `step()` method:**
- Adds `game_features` to info dict when enabled

### 3. Configuration (`src/levd/config.py`)

**New arguments:**
```bash
--use-game-features          # Enable feature prediction (flag)
--game-features "enemy,health"  # Features to predict (comma-separated)
--feature-loss-weight 0.1    # Weight for auxiliary loss
```

### 4. Training Script (`src/levd/run.py`)

**Changes:**
- Parses game features from arguments
- Sets `args.n_features` based on number of features
- Passes `use_labels` and `game_features` to environment kwargs
- Prints enabled features for user feedback

### 5. Algorithm Implementation (`src/levd/algorithm/rainbow.py`)

**Updated `init_network()`:**
- Reads `n_features` from args
- Passes to Rainbow network initialization

### 6. New Utility: Game Features Tracker (`src/levd/game_features.py`)

**Class `GameFeaturesTracker`:**
- Maintains confusion matrix (TP, FP, TN, FN) for each feature
- Methods:
  - `update(predictions, targets)` - Add batch of predictions
  - `get_metrics()` - Compute precision, recall, F1, accuracy
  - `print_stats()` - Display formatted results
  - `reset()` - Clear counters

### 7. Documentation (`src/levd/GAME_FEATURES.md`)

Comprehensive guide including:
- Overview of how it works
- Usage examples
- Architecture diagrams
- Implementation details
- Evaluation metrics
- Comparison with Arnold
- Future enhancement suggestions

## How to Use

### Basic Training Command

```bash
python -m src.levd.run \
    --scenario_name defend_the_center \
    --algorithm rainbow \
    --use-game-features \
    --game-features "enemy,health" \
    --epoch 300
```

### Testing Feature Extraction

```python
import levdoom
from levdoom import Scenario

# Create environment with game features
env = levdoom.make_level(
    Scenario.DEFEND_THE_CENTER, 
    0, 
    use_labels=True,
    game_features=['enemy', 'health']
)[0]

obs, info = env.reset()
obs, reward, done, truncated, info = env.step(0)

# Check if features were extracted
if 'game_features' in info:
    features = info['game_features']
    print(f"Enemy visible: {features[0]}")
    print(f"Health visible: {features[1]}")
```

### Evaluate Feature Predictions

```python
from src.levd.game_features import GameFeaturesTracker
import numpy as np

tracker = GameFeaturesTracker(['enemy', 'health'])

# During evaluation loop
predictions = network_output[0][1]  # Extract features from output
targets = info['game_features']

tracker.update(predictions, targets)

# After evaluation
tracker.print_stats()
metrics = tracker.get_metrics()
```

## Architecture Flow

```
Game State (ViZDoom)
    ↓
Labels Buffer (object segmentation)
    ↓
_extract_game_features()
    ↓
Binary Feature Vector [enemy, health, weapon, ammo]
    ↓
Environment Step Info
    ↓
Network Forward Pass
    ├── Q-Values (main task)
    └── Feature Predictions (auxiliary task)
```

## Key Differences from Arnold

| Aspect | Arnold | LevDoom Implementation |
|--------|--------|----------------------|
| Framework | Custom PyTorch DQN | Tianshou + Gymnasium |
| Loss Integration | Built into DQN training loop | Network ready, policy needs extension |
| Environment | Custom Game class | Gymnasium-based DoomEnv |
| Feature Tracking | GameFeaturesConfusionMatrix | GameFeaturesTracker |
| Training | Fully integrated | Infrastructure ready, loss integration pending |

## Next Steps for Full Integration

To complete the auxiliary loss integration during training:

1. **Create Custom Policy** (`src/levd/algorithm/rainbow_with_features.py`):
   - Extend `RainbowPolicy`
   - Override `learn()` to add feature loss
   - Extract features from batch.info
   - Compute BCE loss and add to total loss

2. **Update Algorithm** (`src/levd/algorithm/rainbow.py`):
   - Use custom policy when features are enabled
   - Add feature loss tracking

3. **Modify Collector**:
   - Ensure `game_features` from info dict are stored in replay buffer
   - May need custom collector or buffer modifications

4. **Add Logging**:
   - Log feature loss to TensorBoard/Wandb
   - Track feature prediction metrics during training
   - Visualize confusion matrices

## Testing

Test the implementation:

```bash
# Test network creation
python -c "
from src.levd.network import Rainbow
import torch

net = Rainbow(
    state_shape=(4, 60, 108),
    action_shape=3,
    n_features=2
)

obs = torch.randn(1, 4, 60, 108)
output, state = net(obs)
print('Output type:', type(output))
if isinstance(output, tuple):
    probs, features = output
    print('Q-probs shape:', probs.shape)
    print('Features shape:', features.shape)
"

# Test feature extraction
python -c "
import levdoom
from levdoom import Scenario

env = levdoom.make_level(Scenario.DEFEND_THE_CENTER, 0, 
                        use_labels=True, 
                        game_features=['enemy', 'health'])[0]
obs, info = env.reset()
obs, r, d, t, info = env.step(0)
print('Game features:', info.get('game_features'))
"
```

## Files Modified

1. ✅ `src/levd/network.py` - Added game features head to Rainbow
2. ✅ `libs/LevDoom/levdoom/envs/base.py` - Added labels and feature extraction
3. ✅ `src/levd/config.py` - Added configuration arguments
4. ✅ `src/levd/run.py` - Integrated features into training setup
5. ✅ `src/levd/algorithm/rainbow.py` - Pass n_features to network
6. ✅ `src/levd/game_features.py` - Created tracker utility (NEW)
7. ✅ `src/levd/GAME_FEATURES.md` - Documentation (NEW)

## Benefits

1. **Improved Learning**: Multi-task learning improves visual representations
2. **Better Generalization**: Features learned transfer to new scenarios
3. **Interpretability**: Can see what agent detects
4. **Regularization**: Prevents overfitting
5. **Research**: Enables study of perception vs. control

## Conclusion

The infrastructure for game features prediction is now in place. The network can predict object presence, the environment provides ground truth labels, and tracking utilities are available. To fully match Arnold's implementation, the remaining step is integrating the auxiliary loss into the training loop via a custom policy class.
