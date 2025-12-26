# A Comparative Analysis of Deep Q-Learning Algorithms in FPS Environment

This repository contains the implementation and experiments for the paper **"A Comparative Analysis between Deep Q-Learning Algorithms in FPS Environment"**, developed as part of the Deep Learning course at the Federal University of Pernambuco (UFPE), Brazil.

## 📄 Paper Abstract

Recent advances in deep reinforcement learning have enabled agents to operate directly from raw visual input in complex environments such as first-person shooter (FPS) games. This work presents a comparative study of value-based deep reinforcement learning architectures in FPS environments using the LevDoom benchmark. We evaluate DQN, C51, Rainbow, DRQN, and Deep Transformer Q-Networks (DTQN) under identical training conditions, with a focus on the **first-ever application of DTQN to FPS environments**. Our results show that attention-based mechanisms (DTQN) achieve K/D ratios up to **40% higher** than standard and recurrent variants on held-out test levels.

## 🎯 Key Contributions

- **First application and evaluation of DTQN** in complex FPS environments
- Empirical comparison of 5 neural architectures under identical training conditions
- Investigation of auxiliary game-feature prediction in Full Deathmatch scenarios
- Analysis of architectural inductive biases for robust policy learning in partially observable environments

## 🏗️ Repository Structure

```
.
├── pretrained/                   # Pre-trained model checkpoints
├── resources/
│   ├── freedoom2.wad            # DOOM resources file (textures, sprites)
│   └── scenarios/               # Game scenario configurations
│       ├── full_deathmatch.wad  # Complete deathmatch maps
│       ├── health_gathering.wad # Simple test scenario
│       └── ...
├── lib/
│   └── LevDoom/                 # Extended LevDoom library
├── src/
│   ├── arnold/                  # Arnold agent implementation
│   ├── levdoom/                 # LevDoom experiments
│   ├── riayn/                   # Rainbow Is All You Need (legacy)
│   └── rainbow/                 # Rainbow implementation (legacy)
├── arnold.py                    # Arnold experiments entry point
├── levd.py                      # LevDoom experiments entry point
├── riayn.py                     # RIAYN experiments (legacy)
├── rainbow.py                   # Rainbow experiments (legacy)
└── README.md
```

## 🧠 Implemented Architectures

### 1. **DQN (Deep Q-Network)**
Standard DQN with experience replay and target network.

```
Conv2d(12→32, 8×8, stride=4) → ReLU →
Conv2d(32→64, 4×4, stride=2) → ReLU →
Conv2d(64→64, 3×3, stride=1) → ReLU →
Flatten → Linear(3136→512) → ReLU → Linear(512→6)
```

### 2. **C51 (Categorical DQN)**
Distributional RL approach modeling the value distribution.

```
[Same CNN encoder as DQN]
Linear(3136→512) → ReLU → Linear(512→306)  # 6 actions × 51 atoms
```

### 3. **Rainbow DQN**
Combines 6 extensions: double DQN, dueling networks, prioritized replay, multi-step learning, distributional RL, and noisy networks.

```
[Shared CNN encoder]
Q-stream: NoisyLinear(3136→512) → ReLU → NoisyLinear(512→306)
V-stream: NoisyLinear(3136→512) → ReLU → NoisyLinear(512→51)
```

### 4. **DRQN (Deep Recurrent Q-Network)**
Adds LSTM for temporal sequence processing.

```
Conv2d(3→32→64→64) → Flatten →
LSTM(3136→512, batch_first=True) →
Linear(512→6)
```

### 5. **DTQN (Deep Transformer Q-Network)** ⭐
**Novel application:** First use of transformer-based attention in FPS environments.

```
[Shared CNN encoder]
Linear(3136→512) [feature projection] →
TransformerDecoder(3 layers, d_model=512, nhead=8, dim_feedforward=2048) →
Linear(512→6)
```

## 🚀 Getting Started

### Installation

```bash
uv sync
```

## 🎮 Running Experiments

### LevDoom Scenarios

#### Training

**Defend the Center:**
```bash
uv run levd.py \
  --algorithm dtqn \
  --scenario defend_the_center \
  --train_levels 0 1 \
  --test_levels 2 3 4 \
  --seed 1 \
  --epoch 50 \
  --lr 0.0001 \
  --step-per-collect 10 \
  --batch-size 64
```

**Full Deathmatch:**
```bash
uv run levd.py \
  --algorithm rainbow \
  --scenario full_deathmatch \
  --train_levels 0 \
  --train_maps 1 2 \
  --test_levels 0 \
  --test_maps 4 11 \
  --seed 42 \
  --epoch 50 \
  --lr 0.0001 \
  --step-per-collect 10 \
  --batch-size 64
```

**Available algorithms:** `dqn`, `c51`, `rainbow`, `drqn`, `dtqn`

#### Testing/Watching

```bash
# Test on specific level
uv run levd.py --watch \
  --algorithm dtqn \
  --scenario defend_the_center \
  --resume-path pretrained/dtqn_best.pth \
  --render \
  --test_levels 4

# Test on specific maps
uv run levd.py --watch \
  --algorithm rainbow \
  --scenario full_deathmatch \
  --resume-path pretrained/rainbow_best.pth \
  --render \
  --test_levels 0 \
  --test_maps 4 11
```

### Arnold Agent

Arnold implementation follows the architecture from [Lample & Chaplot, 2016](https://arxiv.org/abs/1609.05521).

```bash
./run.sh track2 --n_bots 10
```

## 📊 Key Results

### Defend the Center (K/D Ratio on Test Levels)

| Architecture | Level 0 | Level 1 (avg) | Level 2 (avg) | Level 3 (avg) | Level 4 |
|-------------|---------|---------------|---------------|---------------|---------|
| **DTQN** ⭐   | **10.9** | **8.9** | **7.3** | **6.6** | **5.3** |
| Rainbow     | 8.0     | 6.8         | 6.7         | 5.2         | 5.2     |
| DQN         | 6.3     | 6.4         | 6.9         | 5.6         | 5.5     |
| C51         | 6.8     | 6.3         | 6.4         | 5.5         | 5.0     |
| DRQN        | 4.0     | 5.5         | 6.1         | 4.4         | 4.7     |

**DTQN achieves 40% higher K/D ratios** compared to other methods on unseen test levels.

### Full Deathmatch

| Agent | Map 4 K/D | Map 4 Survival | Map 11 K/D | Map 11 Survival |
|-------|-----------|----------------|------------|-----------------|
| Rainbow baseline | 28.9 | 126 | **18.8** | **291** |
| Rainbow + Features | **30.0** | **128** | 17.9 | 169 |

**Key finding:** Auxiliary objectives improve dense combat but don't generalize uniformly across diverse maps.

> **Note:** Full Deathmatch experiments with **Game Feature Prediction** and **Adaptive Movement Reward** are available in the [`matheusvtna/test-game-feat-net`](https://github.com/uandersonricardo/ArnoldSchwarzenegger/tree/matheusvtna/test-game-feat-net) branch.

## 🔬 Experimental Insights

### Why DTQN Outperforms DRQN

1. **Parallel Processing:** Transformers compute representations for all timesteps in parallel vs. sequential LSTM processing
2. **Direct Attention:** Can directly connect firing actions to relevant enemy observations without gradient dilution
3. **No Vanishing Gradients:** Avoids recurrent network training instabilities in off-policy settings
4. **Better Credit Assignment:** Learned positional encodings capture temporal patterns specific to FPS dynamics

### Limitations Discovered

- **Sparse Action Space:** Infrequent enemy appearances make learning correct shooting sequences difficult
- **Computational Budget:** 50 epochs insufficient for full convergence (Arnold required ~60 hours)
- **Pixel-Only Learning:** Limited performance without extensive training or auxiliary objectives
- **Map Diversity:** Auxiliary features can bias agents toward aggressive strategies suboptimal for larger maps

## 📈 Future Work

- [ ] Extend training beyond 100 epochs for full convergence
- [ ] Implement more expressive visual encoders (ResNets, Vision Transformers)
- [ ] Ablation studies of Rainbow's individual components
- [ ] Investigate model-based RL approaches (Dreamer, MuZero)
- [ ] Test in other partially observable domains (StarCraft, Dota 2)

## 🙏 Acknowledgments

- **LevDoom:** Benchmark for studying generalization in RL ([Tomilin et al., 2022](https://arxiv.org/abs/2206.00491))
- **ViZDoom:** Doom-based AI research platform ([Kempka et al., 2016](https://arxiv.org/abs/1605.02097))
- **DTQN:** Deep Transformer Q-Networks ([Esslinger et al., 2022](https://arxiv.org/abs/2206.01078))
- **Arnold:** FPS playing agent ([Lample & Chaplot, 2016](https://arxiv.org/abs/1609.05521))
- **Tianshou:** RL library providing standardized implementations

## 👥 Authors

- **Matheus Andrade** - [GitHub](https://github.com/matheusvtna) - mvtna@cin.ufpe.br
- **Uanderson Silva** - [GitHub](https://github.com/uandersonricardo) - urfs@cin.ufpe.br

**Universidade Federal de Pernambuco (UFPE)**  
Centro de Informática (CIn)
