from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from copy import copy
import os

class EpisodeRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        if 'sc2' in self.args.env:
            self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        else:
            self.env = env_REGISTRY[self.args.env](env_args=self.args.env_args, args=args)

        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        self.mac.mac_plus.init_hidden(batch_size=self.batch_size)

        if self.args.env == 'sc2':
            n_enemy = self.env.n_enemies
            n_agent = self.env.n_agents
            pre_death_enemy = np.zeros(n_enemy)
            pre_enemies_health = np.zeros(n_enemy)
            r_damage_enemies = np.zeros(n_enemy)
            r_kill_enemies = np.zeros(n_enemy)
            r_recover_agents_health = np.zeros(n_agent)
            r_recover_agents_shield = np.zeros(n_agent)
            for t in range(0, n_enemy):
                pre_enemies_health[t] = copy(self.env.enemies[t].health)
            normrate = copy((self.env.max_reward/self.env.reward_scale_rate))
        elif self.args.env == 'stag_hunt':
            n_enemy = 0 
        else:
            n_enemy = 0 

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            # Edited for rd =================================================================================
            if (self.args.env == 'battle_game') or (self.args.env == 'battle_game_state'):
                reward, terminated, env_info, visit_state, actionlist = self.env.step(actions[0].cpu())
            else:
                reward, terminated, env_info = self.env.step(actions[0].cpu())

            if self.args.env == 'battle_game':
                if test_mode==False:
                    if os.path.isdir(self.filepath) ==False :
                        f = open(self.filepath, 'a')
                        f.write(np.array2string(visit_state[0]))
                        f.write('\t')
                        f.write(np.array2string(visit_state[1]))
                        f.write('\t')
                        f.write(np.array2string(actionlist[0]))
                        f.write('\t')
                        f.write(np.array2string(actionlist[1]))
                        f.write('\n')
                        f.close()
                        
            # reward decomposition for each agent ===========================================================
            if self.args.env == 'sc2':
                win_flag = 0
                for t in range(0, n_enemy):
                    r_damage_enemies[t] = 0
                    r_kill_enemies[t] = 0

                r_win = 0
                for t in range(0, n_enemy):
                    if (pre_enemies_health[t]-self.env.enemies[t].health)>0:
                        r_damage_enemies[t] = (pre_enemies_health[t]-self.env.enemies[t].health) / normrate
                    if (self.env.death_tracker_enemy[t] - pre_death_enemy[t]):
                        r_kill_enemies[t] = self.env.reward_death_value / normrate


                for t2 in range(0, n_agent):
                    r_recover_agents_health[t2] = (self.env.previous_ally_units[t2].health-self.env.agents[t2].health)
                    r_recover_agents_shield[t2] = (self.env.previous_ally_units[t2].shield-self.env.agents[t2].shield)
                    r_recover_agents_health[r_recover_agents_health>0] = 0
                    r_recover_agents_shield[r_recover_agents_shield>0] = 0

                    r_recover = ( (r_recover_agents_health.sum()) + (r_recover_agents_shield.sum()) ) / normrate

                if (np.sum(self.env.death_tracker_enemy)==n_enemy):
                    win_flag = 1
                    r_win = self.env.reward_win / normrate
                
                r_plus = r_damage_enemies.sum() + r_kill_enemies.sum() + r_win - r_recover
                r_minus = reward - r_plus

                pre_death_enemy = self.env.death_tracker_enemy.copy()
                for t in range(0, n_enemy):
                    pre_enemies_health[t] = self.env.enemies[t].health
            elif self.args.env == 'stag_hunt':
                n_enemy = 0 
                
                r_minus = env_info['reward_minus'].sum() 
                r_plus = reward - r_minus
                if r_plus > 0 and r_minus < 0:
                    n_enemy = 0
                if r_minus < 0:
                    n_enemy = 0
            elif self.args.env == 'battle_game':
                r_plus = env_info['r_plus']
                r_minus = env_info['r_minus']
            elif self.args.env == 'battle_game_state':
                r_plus = env_info['r_plus']
                r_minus = env_info['r_minus']
            elif self.args.env == 'mtgame':
                r_plus = env_info['r_plus']
                r_minus = env_info['r_minus']
            # ======================================================================================================


            #reward, terminated, env_info = self.env.step(actions[0].cpu())
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "r_plus": [(r_plus,)],
                "r_minus": [(r_minus,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

        

        self.batch.update({"actions": actions}, ts=self.t)

        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)

        if test_mode and (len(self.test_returns) == self.args.test_nepisode):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()
