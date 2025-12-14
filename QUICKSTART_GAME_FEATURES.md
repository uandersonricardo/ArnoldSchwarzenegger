# Quick Start: Game Features Prediction

## What is it?
Game features prediction is an auxiliary task that helps the AI agent learn to detect objects (enemies, health, weapons, ammo) in the game, improving its visual understanding and performance.

## Quick Usage

### 1. Basic Training
```bash
python -m src.levd.run \
    --scenario_name defend_the_center \
    --algorithm rainbow \
    --use-game-features \
    --game-features "enemy,health"
```

### 2. All Features
```bash
python -m src.levd.run \
    --scenario_name full_deathmatch \
    --algorithm rainbow \
    --use-game-features \
    --game-features "enemy,health,weapon,ammo" \
    --feature-loss-weight 0.15
```

### 3. In Code
```python
import levdoom
from levdoom import Scenario

# Create environment
env = levdoom.make_level(
    Scenario.DEFEND_THE_CENTER, 0,
    use_labels=True,
    game_features=['enemy', 'health']
)[0]

# Use it
obs, info = env.reset()
obs, r, done, truncated, info = env.step(action)

# Check features
if 'game_features' in info:
    enemy_visible = info['game_features'][0]
    health_visible = info['game_features'][1]
```

## Available Features
- `enemy` - Monsters and enemy marines
- `health` - Health pickups and armor
- `weapon` - Weapon pickups
- `ammo` - Ammunition pickups

## Key Files
- Network: `src/levd/network.py` (Rainbow class)
- Environment: `libs/LevDoom/levdoom/envs/base.py`
- Config: `src/levd/config.py`
- Tracker: `src/levd/game_features.py`
- Examples: `examples/game_features_example.py`
- Full docs: `src/levd/GAME_FEATURES.md`

## Run Example
```bash
python examples/game_features_example.py
```

## Benefits
✅ Better visual learning
✅ Improved performance
✅ More interpretable agent
✅ Better generalization

## Need Help?
See `src/levd/GAME_FEATURES.md` for detailed documentation.
