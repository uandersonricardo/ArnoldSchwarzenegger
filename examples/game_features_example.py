#!/usr/bin/env python3
"""
Example script demonstrating game features prediction with LevDoom.

This script shows how to:
1. Create an environment with game features enabled
2. Train a Rainbow agent with auxiliary feature prediction
3. Track and evaluate feature prediction accuracy
"""

import numpy as np
import torch
from argparse import Namespace

# Example 1: Create environment with game features
def create_env_with_features():
    """Create a LevDoom environment with game features enabled."""
    import levdoom
    from levdoom import Scenario
    
    print("Creating environment with game features...")
    
    env, _ = levdoom.make_level(
        Scenario.DEFEND_THE_CENTER,
        level=0,
        use_labels=True,
        game_features=['enemy', 'health']
    )
    
    print(f"Environment created: {env.name}")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Reset and take a step
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    
    if 'game_features' in info:
        features = info['game_features']
        print(f"\nGame features detected:")
        print(f"  Enemy visible: {bool(features[0])}")
        print(f"  Health visible: {bool(features[1])}")
    else:
        print("No game features in info (this is expected if labels buffer is empty)")
    
    env.close()
    return True


# Example 2: Create network with game features
def create_network_with_features():
    """Create a Rainbow network with game features prediction head."""
    from src.levd.network import Rainbow
    
    print("\nCreating Rainbow network with game features...")
    
    # Network configuration
    state_shape = (4, 60, 108)  # (frames, height, width)
    action_shape = 3
    n_features = 2  # enemy, health
    
    net = Rainbow(
        state_shape=state_shape,
        action_shape=action_shape,
        num_atoms=51,
        n_features=n_features,
        hidden_dim=512,
        device='cpu'
    )
    
    print(f"Network created with {n_features} game features")
    
    # Test forward pass
    dummy_obs = torch.randn(1, *state_shape)
    output, state = net(dummy_obs)
    
    if isinstance(output, tuple):
        q_probs, features = output
        print(f"\nForward pass successful:")
        print(f"  Q-value probs shape: {q_probs.shape}")
        print(f"  Game features shape: {features.shape}")
        print(f"  Feature predictions (0-1): {features[0].detach().numpy()}")
    else:
        print(f"\nQ-value output shape: {output.shape}")
    
    return net


# Example 3: Track feature predictions
def track_feature_predictions():
    """Demonstrate game features tracking during evaluation."""
    from src.levd.game_features import GameFeaturesTracker
    
    print("\n" + "="*70)
    print("Game Features Tracking Example")
    print("="*70)
    
    # Initialize tracker
    tracker = GameFeaturesTracker(['enemy', 'health'])
    
    # Simulate some predictions and ground truth
    np.random.seed(42)
    n_samples = 100
    
    for i in range(n_samples):
        # Simulate predictions (0-1 probabilities)
        predictions = np.random.rand(2)
        
        # Simulate ground truth (binary)
        # Let's say enemy is present 60% of time, health 20%
        targets = np.array([
            1.0 if np.random.rand() < 0.6 else 0.0,  # enemy
            1.0 if np.random.rand() < 0.2 else 0.0   # health
        ])
        
        # Update tracker
        tracker.update(predictions, targets)
    
    # Print statistics
    tracker.print_stats()
    
    # Get metrics programmatically
    metrics = tracker.get_metrics()
    for feature_name, m in metrics.items():
        print(f"{feature_name}: F1={m['f1']:.3f}, Accuracy={m['accuracy']:.3f}")


# Example 4: Training command
def print_training_command():
    """Print example training command with game features."""
    print("\n" + "="*70)
    print("Training with Game Features")
    print("="*70)
    
    command = """
# Train Rainbow on Defend the Center with enemy and health prediction
python -m src.levd.run \\
    --scenario_name defend_the_center \\
    --algorithm rainbow \\
    --use-game-features \\
    --game-features "enemy,health" \\
    --feature-loss-weight 0.1 \\
    --epoch 300 \\
    --train_levels 0 1 \\
    --test_levels 2 3 4 \\
    --device cuda \\
    --seed 42

# Train with all feature types
python -m src.levd.run \\
    --scenario_name full_deathmatch \\
    --algorithm rainbow \\
    --use-game-features \\
    --game-features "enemy,health,weapon,ammo" \\
    --feature-loss-weight 0.1 \\
    --epoch 500 \\
    --training-num 10 \\
    --batch-size 256
"""
    print(command)


def main():
    """Run all examples."""
    print("="*70)
    print("LevDoom Game Features Prediction - Examples")
    print("="*70)
    
    try:
        # Example 1: Environment
        create_env_with_features()
        
        # Example 2: Network
        create_network_with_features()
        
        # Example 3: Tracking
        track_feature_predictions()
        
        # Example 4: Training command
        print_training_command()
        
        print("\n" + "="*70)
        print("All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
