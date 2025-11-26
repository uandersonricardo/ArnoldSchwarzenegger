# LEVDOOM

Treino
uv run levd.py --algorithm dqn --scenario defend_the_center --train_levels 0 1 --test_levels 2 3 4 --seed 1 --epoch 100 --lr 0.0001 --step-per-collect 10 --batch-size 64

uv run levd.py --algorithm dqn --scenario defend_the_center --train_levels 0 --train_maps 1 2 --test_levels 0 --test_maps 3 4 5 --seed 42 --epoch 20 --lr 0.0001 --step-per-collect 10 --batch-size 64

Epoch #1: 100001it [04:09, 400.13it/s, env_step=100000, len=118, loss=0.136, n/ep=0, n/st=16, rew=2.66]                                                                       
Epoch #1: test_reward: 3.439600 ± 1.981931, best_reward: 3.439600 ± 1.981931 in #1
Epoch #2: 100001it [04:09, 400.73it/s, env_step=200000, len=71, loss=0.131, n/ep=0, n/st=16, rew=1.79]                                                                        
Epoch #2: test_reward: 4.225500 ± 1.894355, best_reward: 4.225500 ± 1.894355 in #2
Epoch #3: 100001it [04:10, 399.98it/s, env_step=300000, len=101, loss=0.103, n/ep=0, n/st=16, rew=2.51]                                                                       
Epoch #3: test_reward: 4.621900 ± 2.267661, best_reward: 4.621900 ± 2.267661 in #3
Epoch #4: 100001it [04:09, 400.32it/s, env_step=400000, len=107, loss=0.091, n/ep=0, n/st=16, rew=3.71]                                                                       
Epoch #4: test_reward: 3.950800 ± 2.029332, best_reward: 4.621900 ± 2.267661 in #3
Epoch #5: 100001it [04:10, 399.89it/s, env_step=500000, len=191, loss=0.080, n/ep=0, n/st=16, rew=6.68]                                                                       
Epoch #5: test_reward: 5.041700 ± 1.853443, best_reward: 5.041700 ± 1.853443 in #5
Epoch #6: 100001it [04:07, 403.31it/s, env_step=600000, len=122, loss=0.074, n/ep=0, n/st=16, rew=4.67]                                                                       
Epoch #6: test_reward: 5.109000 ± 1.928644, best_reward: 5.109000 ± 1.928644 in #6
Epoch #7: 100001it [05:11, 320.84it/s, env_step=700000, len=83, loss=0.067, n/ep=0, n/st=16, rew=2.55]                                                                        
Epoch #7: test_reward: 5.054600 ± 2.263414, best_reward: 5.109000 ± 1.928644 in #6
Epoch #8: 100001it [05:42, 291.85it/s, env_step=800000, len=138, loss=0.063, n/ep=0, n/st=16, rew=5.70]                                                                       
Epoch #8: test_reward: 5.533000 ± 1.941397, best_reward: 5.533000 ± 1.941397 in #8
Epoch #9: 100001it [05:23, 308.86it/s, env_step=900000, len=180, loss=0.063, n/ep=0, n/st=16, rew=6.67]                                                                       
Epoch #9: test_reward: 5.613200 ± 1.675593, best_reward: 5.613200 ± 1.675593 in #9

Teste
uv run levdw.py --algorithm dqn --scenario full_deathmatch --resume-path 1_20251123_231836/policy_best.pth --watch --render --test_levels 1

uv run levdw.py --algorithm dqn --scenario full_deathmatch --resume-path 1_20251123_231836/policy_best.pth --watch --render --test_levels 0 --test_maps 2 4

uv run levdw.py --algorithm dqn --scenario full_deathmatch --resume-path 1_20251123_231836/policy_best.pth --watch --render --test_levels 0 --test_maps 1 2 3 4 5

# Arnold

Arnold is a PyTorch implementation of the agent presented in *Playing FPS Games with Deep Reinforcement Learning* (https://arxiv.org/abs/1609.05521), and that won the 2017 edition of the [*ViZDoom AI Competition*](http://vizdoom.cs.put.edu.pl/competition-cig-2017).

![example](./docs/example.gif) 

### This repository contains:
- The source code to train DOOM agents
- A package with 17 selected maps that can be used for training and evaluation
- 5 pretrained models that you can visualize and play against, including the ones that won the ViZDoom competition


## Installation

#### Dependencies
Arnold was tested successfully on Mac OS and Linux distributions. You will need:
- Python 2/3 with NumPy and OpenCV
- PyTorch
- ViZDoom

Follow the instructions on https://github.com/mwydmuch/ViZDoom to install ViZDoom. Be sure that you can run `import vizdoom` in Python from any directory. To do so, you can either install the library with `pip`, or compile it, then move it to the `site-packages` directory of your Python installation, as explained here: https://github.com/mwydmuch/ViZDoom/blob/master/doc/Quickstart.md.


## Code structure

    .
    ├── pretrained                    # Examples of pretrained models
    ├── resources
        ├── freedoom2.wad             # DOOM resources file (containing all textures)
        └── scenarios                 # Folder containing all scenarios
            ├── full_deathmatch.wad   # Scenario containing all deathmatch maps
            ├── health_gathering.wad  # Simple test scenario
            └── ...
    ├── src                           # Source files
        ├── doom                      # Game interaction / API / scenarios
        ├── model                     # DQN / DRQN implementations
        └── trainer                   # Folder containing training scripts
    ├── arnold.py                     # Main file
    └── README.md


## Scenarios / Maps

## Train a model

There are many parameters you can tune to train a model.


```bash
python arnold.py

## General parameters about the game
--freedoom "true"                # use freedoom resources
--height 60                      # screen height
--width 108                      # screen width
--gray "false"                   # use grayscale screen
--use_screen_buffer "true"       # use the screen buffer (what the player sees)
--use_depth_buffer "false"       # use the depth buffer
--labels_mapping ""              # use extra feature maps for specific objects
--game_features "target,enemy"   # game features prediction (auxiliary tasks)
--render_hud "false"             # render the HUD (status bar in the bottom of the screen)
--render_crosshair "true"        # render crosshair (targeting aid in the center of the screen)
--render_weapon "true"           # render weapon
--hist_size 4                    # history size
--frame_skip 4                   # frame skip (1 = keep every frame)

## Agent allowed actions
--action_combinations "attack+move_lr;turn_lr;move_fb"  # agent allowed actions
--freelook "false"               # allow the agent to look up and down
--speed "on"                     # make the agent run
--crouch "off"                   # make the agent crouch

## Training parameters
--batch_size 32                  # batch size
--replay_memory_size 1000000     # maximum number of frames in the replay memory
--start_decay 0                  # epsilon decay iteration start
--stop_decay 1000000             # epsilon decay iteration end
--final_decay 0.1                # final epsilon value
--gamma 0.99                     # discount factor gamma
--dueling_network "false"        # use a dueling architecture
--clip_delta 1.0                 # clip the delta loss
--update_frequency 4             # DQN update frequency
--dropout 0.5                    # dropout on CNN output layer
--optimizer "rmsprop,lr=0.0002"  # network optimizer

## Network architecture
--network_type "dqn_rnn"         # network type (dqn_ff / dqn_rnn)
--recurrence "lstm"              # recurrent network type (rnn / gru / lstm)
--n_rec_layers 1                 # number of layers in the recurrent network
--n_rec_updates 5                # number of updates by sample
--remember 1                     # remember all frames during evaluation
--use_bn "off"                   # use BatchNorm when processing the screen
--variable_dim "32"              # game variables embeddings dimension
--bucket_size "[10, 1]"          # bucket game variables (typically health / ammo)
--hidden_dim 512                 # hidden layers dimension

## Scenario parameters (these parameters will differ based on the scenario)
--scenario "deathmatch"          # scenario
--wad "full_deathmatch"          # WAD file (scenario file)
--map_ids_train "2,3,4,5"        # maps to train the model
--map_ids_test "6,7,8"           # maps to test the model
--n_bots 8                       # number of enemy bots
--randomize_textures "true"      # randomize walls / floors / ceils textures during training
--init_bots_health 20            # reduce initial life of enemy bots (helps a lot when using pistol)

## Various
--exp_name new_train             # experiment name
--dump_freq 200000               # periodically dump the model
--gpu_id -1                      # GPU ID (-1 to run on CPU)
```

Once your agent is trained, you can visualize it by running the same command, and using the following extra arguments:
```bash
--visualize 1                    # visualize the model (render the screen)
--evaluate 1                     # evaluate the agent
--manual_control 1               # manually make the agent turn about when it gets stuck
--reload PATH                    # path where the trained agent was saved
```


Here are some examples of training commands for 3 different scenarios:

#### Defend the center

In this scenario the agent is in the middle of a circular map. Monsters are regularly appearing on the sides and are walking towards the agent. The agent is given a pistol and limited ammo, and must turn around and kill the monsters before they reach it. The following command trains a standard DQN, that should reach the optimal performance of 56 frags (the number of bullets in the pistol) in about 4 million steps:

```bash
python arnold.py --scenario defend_the_center --action_combinations "turn_lr+attack" --frame_skip 2
```

#### Health gathering

In this scenario the agent is walking on lava, and is losing health points at each time step. The agent has to move and collect as many health pack as possible in order to survive. The objective is to survive the longest possible time.

```bash
python arnold.py --scenario health_gathering --action_combinations "move_fb;turn_lr" --frame_skip 5
```

This scenario is very easy and the model quickly reaches the maximum survival time of 2 minutes (35 * 120 = 4200 frames). The scenario also provides a `supreme` mode, in which the map is more complicated and where the health packs are much harder to collect:

```bash
python arnold.py --scenario health_gathering --action_combinations "move_fb;turn_lr" --frame_skip 5 --supreme 1
```

In this scenario, the agent takes about 1.5 million steps to reach the maximum survival time (but often dies before the end).

#### Deathmatch

In this scenario, the agent is trained to fight against the built-in bots of the game. Here is a command to train the agent using game features prediction (as described in [1]), and a DRQN:

```bash
python arnold.py --scenario deathmatch --wad deathmatch_rockets --n_bots 8 \
--action_combinations "move_fb;move_lr;turn_lr;attack" --frame_skip 4 \
--game_features "enemy" --network_type dqn_rnn --recurrence lstm --n_rec_updates 5
```


## Pretrained models

#### Defend the center / Health gathering

We provide a pretrained model for each of these scenarios. You can visualize them by running:

```bash
./run.sh defend_the_center
```

or

```bash
./run.sh health_gathering
```

#### Visual Doom AI Competition 2017

We release the two agents submitted to the first and second tracks of the ViZDoom AI 2017 Competition. You can visualize them playing against the built-in bots using the following commands:

##### Track 1 - Arnold vs 10 built-in AI bots
```bash
./run.sh track1 --n_bots 10
```

##### Track 2 - Arnold vs 10 built-in AI bots - Map 2
```bash
./run.sh track2 --n_bots 10 --map_id 2
```

##### Track 2 - 4 Arnold playing against each other - Map 3
```bash
./run.sh track2 --n_bots 0 --map_id 3 --n_agents 4
```

We also trained an agent on a single map, using a same weapon (the SuperShotgun). This agent is extremely difficult to beat.

##### Shotgun - 4 Arnold playing against each other
```bash
./run.sh shotgun --n_bots 0 --n_agents 4
```

##### Shotgun - 3 Arnold playing against each other + 1 human player (to play against the agent)
```bash
./run.sh shotgun --n_bots 0 --n_agents 3 --human_player 1
```


## References

If you found this code useful, please consider citing:

[1] G. Lample\* and D.S. Chaplot\*, [*Playing FPS Games with Deep Reinforcement Learning*](https://arxiv.org/abs/1609.05521)
```
@inproceedings{lample2017playing,
  title={Playing FPS Games with Deep Reinforcement Learning.},
  author={Lample, Guillaume and Chaplot, Devendra Singh},
  booktitle={Proceedings of AAAI},
  year={2017}
}
```


[2] D.S. Chaplot\* and G. Lample\*, [*Arnold: An Autonomous Agent to Play FPS Games*](http://www.cs.cmu.edu/~dchaplot/papers/arnold_aaai17.pdf)
```
@inproceedings{chaplot2017arnold,
  title={Arnold: An Autonomous Agent to Play FPS Games.},
  author={Chaplot, Devendra Singh and Lample, Guillaume},
  booktitle={Proceedings of AAAI},
  year={2017},
  Note={Best Demo award}
}
```

## Acknowledgements
We acknowledge the developers of [*ViZDoom*](http://vizdoom.cs.put.edu.pl/) for constant help and support during the development of this project. Some of the maps and wad files have been borrowed from the ViZDoom [*git repository*](https://github.com/mwydmuch/ViZDoom). We also thank the members of the [*ZDoom*](https://forum.zdoom.org/) community for their help with the Action Code Scripts (ACS).
