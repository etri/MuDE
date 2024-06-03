# MuDE: Multi-agent Decomposed Reward-based Exploration

## Note

 This codebase accompanies paper *MuDE: Multi-agent decomposed reward-based exploration*, 
 and is based on  [PyMARL](https://github.com/oxwhirl/pymarl) and [SMAC](https://github.com/oxwhirl/smac) codebases which are open-sourced. 

The implementation of the following methods can also be found in this codebase, which are finished by the authors of following papers:

- [**QPLEX**: QPLEX: Duplex Dueling Multi-Agent Q-Learning](https://arxiv.org/abs/2008.01062)
- [**QMIX**: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485)
- [**RD<sup>2</sup>**: Reward Decomposition with Representation Decomposition](https://proceedings.neurips.cc/paper/2020/hash/82039d16dce0aab3913b6a7ac73deff7-Abstract.html)

Please refer to [PyMARL](https://github.com/oxwhirl/pymarl) for setting up the experimental environment.



## Run an experiment 

## SMAC

```shell
python3 src/main.py --config=mude --env-config=sc2 with env_args.map_name=3m
```

## Modified Predator-prey (MPP)

```shell
python3 src/main.py --config=mude --env-config=pred_prey_punish
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

All results will be stored in the `Results` folder.



## Saving and loading learnt models

### Saving models

You can save the learnt models to disk by setting `save_model = True`, which is set to `False` by default. The frequency of saving models can be adjusted using `save_model_interval` configuration. Models will be saved in the result directory, under the folder called *models*. The directory corresponding each run will contain models saved throughout the experiment, each within a folder corresponding to the number of timesteps passed since starting the learning process.

### Loading models

Learnt models can be loaded using the `checkpoint_path` parameter, after which the learning will proceed from the corresponding timestep. 

## Watching StarCraft II replays

`save_replay` option allows saving replays of models which are loaded using `checkpoint_path`. Once the model is successfully loaded, `test_nepisode` number of episodes are run on the test mode and a .SC2Replay file is saved in the Replay directory of StarCraft II. Please make sure to use the episode runner if you wish to save a replay, i.e., `runner=episode`. The name of the saved replay file starts with the given `env_args.save_replay_prefix` (map_name if empty), followed by the current timestamp. 

The saved replays can be watched by double-clicking on them or using the following command:

```shell
python -m pysc2.bin.play --norender --rgb_minimap_size 0 --replay NAME.SC2Replay
```

**Note:** Replays cannot be watched using the Linux version of StarCraft II. Please use either the Mac or Windows version of the StarCraft II client.

