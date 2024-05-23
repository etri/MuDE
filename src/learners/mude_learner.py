# From https://github.com/wjh720/QPLEX/, added here for convenience.
import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.dmaq_general import DMAQer
from modules.mixers.dmaq_qatten import DMAQ_QattenMixer
from modules.predict.mask_state import MaskFuncGlobal
from modules.predict.mask_state_large import MaskFuncGlobalLarge
from modules.predict.sub_reward_func import SubRewardFunc
from modules.predict.sub_reward_func_pos import SubRewardFuncPos
from modules.predict.sub_reward_func_neg import SubRewardFuncNeg
from modules.predict.full_reward_func import FullRewardFunc
from modules.predict.reward_classifier import RewardClassifier
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from torch.optim import RMSprop
import numpy as np


class MUDE_QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.mac_plus = copy.deepcopy(self.mac)
        self.params += list(self.mac_plus.parameters())

        self.MaskPlus = MaskFuncGlobal(args)
        self.MaskMinus = MaskFuncGlobal(args)
        self.params += list(self.MaskPlus.parameters())
        self.params += list(self.MaskMinus.parameters())

        self.sub_reward_func_pos = SubRewardFuncPos(scheme, args)
        self.sub_reward_func_neg = SubRewardFuncNeg(scheme, args)
        self.reward_classify = RewardClassifier(scheme, args)

        self.params += list(self.sub_reward_func_pos.parameters())
        self.params += list(self.sub_reward_func_neg.parameters())
        self.params += list(self.reward_classify.parameters())

        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "dmaq":
                self.mixer = DMAQer(args)
                self.mixer_plus = DMAQer(args)
            elif args.mixer == 'dmaq_qatten':
                self.mixer = DMAQ_QattenMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.params += list(self.mixer_plus.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)
            self.target_mixer_plus = copy.deepcopy(self.mixer_plus)

        self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.target_mac_plus = copy.deepcopy(self.mac_plus)
        self.mac.mac_plus = self.mac_plus

        self.log_stats_t = -self.args.learner_log_interval - 1

        self.n_actions = self.args.n_actions
        self.maskplusinput = self.args.initmask * th.ones(self.args.state_shape).cuda()  
        self.maskminusinput = self.args.initmask * th.ones(self.args.state_shape).cuda() 

    def sub_train(self, batch: EpisodeBatch, t_env: int, episode_num: int, mac, mixer, optimiser, params,
                  show_demo=False, save_data=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        r_plus = batch["r_plus"][:, :-1]
        r_minus = batch["r_minus"][:, :-1] 
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        actions_onehot = batch["actions_onehot"][:, :-1]

        # Calculate estimated Q-Values
        mac_out = []
        mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out = mac_out.clone().detach()
        x_mac_out[avail_actions == 0] = -9999999
        max_action_qvals, max_action_index = x_mac_out[:, :-1].max(dim=3)

        max_action_index = max_action_index.detach().unsqueeze(3)
        is_max_action = (max_action_index == actions).int().float()

        if show_demo:
            q_i_data = chosen_action_qvals.detach().cpu().numpy()
            q_data = (max_action_qvals - chosen_action_qvals).detach().cpu().numpy()

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999


        # Calculate about Q+ ========================================================================================
        # Calculate estimated Q-Values
        mac_out_plus = []
        self.mac_plus.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs_plus = self.mac_plus.forward(batch, t=t)
            mac_out_plus.append(agent_outs_plus)
        mac_out_plus = th.stack(mac_out_plus, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals_plus = th.gather(mac_out_plus[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        x_mac_out_plus = mac_out_plus.clone().detach()
        x_mac_out_plus[avail_actions == 0] = -9999999
        max_action_qvals_plus, max_action_index_plus = x_mac_out_plus[:, :-1].max(dim=3)

        # Calculate the Q-Values necessary for the target
        target_mac_out_plus = []
        self.target_mac_plus.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs_plus = self.target_mac_plus.forward(batch, t=t)
            target_mac_out_plus.append(target_agent_outs_plus)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out_plus = th.stack(target_mac_out_plus[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out_plus[avail_actions[:, 1:] == 0] = -9999999  # From OG deepmarl
        # ============================================================================================================

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            target_max_qvals = target_mac_out.max(dim=3)[0]
            target_next_actions = cur_max_actions.detach()

            cur_max_actions_onehot = th.zeros(cur_max_actions.squeeze(3).shape + (self.n_actions,)).cuda()
            cur_max_actions_onehot = cur_max_actions_onehot.scatter_(3, cur_max_actions, 1)
        else:
            # Calculate the Q-Values necessary for the target
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)
            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time
            target_max_qvals = target_mac_out.max(dim=3)[0]

        # Calculate about Q+ ========================================================================================
        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach_plus = mac_out_plus.clone().detach()
            mac_out_detach_plus[avail_actions == 0] = -9999999
            cur_max_actions_plus = mac_out_detach_plus[:, 1:].max(dim=3, keepdim=True)[1]
            target_chosen_qvals_plus = th.gather(target_mac_out_plus, 3, cur_max_actions_plus).squeeze(3)
            target_max_qvals_plus = target_mac_out_plus.max(dim=3)[0]
            target_next_actions_plus = cur_max_actions_plus.detach()

            cur_max_actions_onehot_plus = th.zeros(cur_max_actions_plus.squeeze(3).shape + (self.n_actions,)).cuda()
            cur_max_actions_onehot_plus = cur_max_actions_onehot_plus.scatter_(3, cur_max_actions_plus, 1)
        # ============================================================================================================

        # Mix
        if mixer is not None:
            ans_chosen = mixer(chosen_action_qvals, batch["state"][:, :-1], is_v=True)
            ans_adv = mixer(chosen_action_qvals, batch["state"][:, :-1], actions=actions_onehot,
                            max_q_i=max_action_qvals, is_v=False)
            chosen_action_qvals = ans_chosen + ans_adv

            ans_chosen_plus = self.mixer_plus(chosen_action_qvals_plus, batch["state"][:, :-1], is_v=True)
            ans_adv_plus = self.mixer_plus(chosen_action_qvals_plus, batch["state"][:, :-1], actions=actions_onehot,
                            max_q_i=max_action_qvals_plus, is_v=False)
            chosen_action_qvals_plus = ans_chosen_plus + ans_adv_plus

            if self.args.double_q:
                target_chosen = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:], is_v=True)
                target_adv = self.target_mixer(target_chosen_qvals, batch["state"][:, 1:],
                                                actions=cur_max_actions_onehot,
                                                max_q_i=target_max_qvals, is_v=False)
                target_max_qvals = target_chosen + target_adv
                
                target_chosen_plus = self.target_mixer_plus(target_chosen_qvals_plus, batch["state"][:, 1:], is_v=True)
                target_adv_plus = self.target_mixer_plus(target_chosen_qvals_plus, batch["state"][:, 1:],
                                                actions=cur_max_actions_onehot_plus ,
                                                max_q_i=target_max_qvals_plus , is_v=False)
                target_max_qvals_plus = target_chosen_plus + target_adv_plus

            else:
                target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:], is_v=True)
                target_max_qvals_plus = self.target_mixer2(target_max_qvals_plus, batch["state"][:, 1:], is_v=True)

        # Reward classifier loss =====================================================================================
        mask_ex = mask.expand_as(rewards)
        est_cls = self.reward_classify.forward(batch)
        target_cls = th.where(rewards<0, th.ones_like(rewards), th.zeros_like(rewards))
        masked_est_cls = est_cls * mask_ex
        masked_target_cls = target_cls * mask_ex
        celoss_func = nn.BCELoss()
        celoss = celoss_func(masked_est_cls, masked_target_cls)
        # ============================================================================================================

        # Find minimum or maximum sub-reward samples =================================================================
        onpos = th.where(r_plus==0, th.zeros_like(r_plus), th.ones_like(r_plus))
        onneg = th.where(r_minus==0, th.zeros_like(r_minus), th.ones_like(r_minus))
        onreward = onpos * onneg # pos, neg reward가 모두 0이 아닌 데이터 index
        zeroreward = th.where(rewards==0, th.ones_like(rewards), th.zeros_like(rewards)) # reward가 0인 데이터
        target_nonzeroreward = zeroreward * onreward # reward가 0이면서 pos, neg reward가 0이 아닌 데이터 index

        predicted_class = th.where(est_cls>=0.5, th.ones_like(rewards), th.zeros_like(rewards))
        #no_reward_class = th.where(est_cls< 0.5, th.ones_like(rewards), th.zeros_like(rewards))
        no_reward_class = th.where(est_cls< self.args.cls_crietrion, th.ones_like(rewards), th.zeros_like(rewards))
        masked_error_noreward = (target_nonzeroreward - predicted_class) * zeroreward * mask_ex
        acc_norewardsample_cls = (th.abs(masked_error_noreward)).sum() / (zeroreward * mask_ex).sum()

        masked_nonzeroreward = target_nonzeroreward * mask_ex
        acc_norewardsample_ratio = (masked_nonzeroreward).sum() / (zeroreward * mask_ex).sum()

        # Sub-reward network training=================================================================================
        maskplus = self.MaskPlus(self.maskplusinput)
        maskminus = self.MaskMinus(self.maskminusinput)
        pos_rewards = self.sub_reward_func_pos.forward(batch, maskplus)
        neg_rewards = self.sub_reward_func_neg.forward(batch, maskminus)
        est_rewards = (pos_rewards + neg_rewards)



        r_error = (est_rewards - rewards.clone().detach())
        mask_reward = mask.expand_as(r_error)
        if self.args.rerror_mag == 1:
            masked_r_error = r_error * mask_reward * th.abs(rewards)
        else:
            masked_r_error = r_error * mask_reward
        r_loss = (masked_r_error ** 2).sum() / mask_reward.sum()
        r_plus_mse = ( ( (r_plus - pos_rewards)* mask_reward) ** 2).sum() / (mask_reward).sum()
        r_minus_mse = ( ( (r_minus - neg_rewards)* mask_reward) ** 2).sum() / (mask_reward).sum()

        r_zero_loss_plus = ((pos_rewards*no_reward_class*zeroreward) ** 2).sum() / (mask_reward).sum()
        r_zero_loss_minus = ((neg_rewards*no_reward_class*zeroreward) ** 2).sum() / (mask_reward).sum()

        #delval = 1e-7 
        if self.args.delval == 1:
            delval = 1e-15
        else:
            delval = 1e-7
        #delval = 1e-15
        Lmini = th.log(maskplus.sum()+delval) + th.log(maskminus.sum()+delval)
        Ldiv1_12 = -(th.log(F.relu(maskplus - maskminus).sum()+delval))
        Ldiv1_21 = -(th.log(F.relu(maskminus - maskplus).sum()+delval))
        Ldiv1 = (Ldiv1_12 + Ldiv1_21).sum()

        # ============================================================================================================

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        targets2 = pos_rewards + self.args.gamma * (1 - terminated) * target_max_qvals_plus

        if show_demo:
            tot_q_data = chosen_action_qvals.detach().cpu().numpy()
            tot_target = targets.detach().cpu().numpy()
            print('action_pair_%d_%d' % (save_data[0], save_data[1]), np.squeeze(q_data[:, 0]),
                  np.squeeze(q_i_data[:, 0]), np.squeeze(tot_q_data[:, 0]), np.squeeze(tot_target[:, 0]))
            self.logger.log_stat('action_pair_%d_%d' % (save_data[0], save_data[1]),
                                 np.squeeze(tot_q_data[:, 0]), t_env)
            return

        # Td-error
        td_error1 = (chosen_action_qvals - targets.detach())
        td_error2 = (chosen_action_qvals_plus - targets2.detach())

        mask1 = mask.expand_as(td_error1)
        mask2 = mask.expand_as(td_error2)

        # 0-out the targets that came from padded data
        masked_td_error1 = td_error1 * mask1
        masked_td_error2 = td_error2 * mask2

        loss1 = (masked_td_error1 ** 2).sum() / mask1.sum()
        loss2 = (masked_td_error2 ** 2).sum() / mask2.sum()

        masked_hit_prob = th.mean(is_max_action, dim=2) * mask
        hit_prob = masked_hit_prob.sum() / mask.sum()
        
        if self.args.otherlossoff == 1:
            if t_env > self.args.Act_Decay_End:
                self.args.q_loss2 = 0.0
                self.args.r_loss = 0.0
                self.args.Lmini = 0.0
                self.args.Ldiv1 = 0.0 
                self.args.r_zero_loss_plus = 0.0
                self.args.r_zero_loss_minus = 0.0
                self.args.celoss = 0.0

        loss =  self.args.q_loss * loss1 + self.args.q_loss2* loss2 + self.args.r_loss * r_loss + self.args.Lmini * Lmini  + self.args.Ldiv1 * Ldiv1 + \
          self.args.celoss * celoss  + self.args.r_zero_loss_plus* r_zero_loss_plus + self.args.r_zero_loss_minus* r_zero_loss_minus

        r_mean = th.abs(th.mean(rewards))
        r_std = th.std(rewards)
        r_range = th.max(rewards) - th.min(rewards)

        optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(params, self.args.grad_norm_clip)
        optimiser.step()

        stdval1 = np.sqrt(np.cov(th.flatten(pos_rewards).cpu().detach().numpy(), th.flatten(r_plus).cpu().detach().numpy())[0,0])
        stdval2 = np.sqrt(np.cov(th.flatten(pos_rewards).cpu().detach().numpy(), th.flatten(r_plus).cpu().detach().numpy())[1,1])
        cov1 = np.cov(th.flatten(pos_rewards).cpu().detach().numpy(), th.flatten(r_plus).cpu().detach().numpy())[0,1]
        corr_plus = cov1 / (stdval1 * stdval2)
        stdval3 = np.sqrt(np.cov(th.flatten(neg_rewards).cpu().detach().numpy(), th.flatten(r_minus).cpu().detach().numpy())[0,0])
        stdval4 = np.sqrt(np.cov(th.flatten(neg_rewards).cpu().detach().numpy(), th.flatten(r_minus).cpu().detach().numpy())[1,1])
        cov3 = np.cov(th.flatten(neg_rewards).cpu().detach().numpy(), th.flatten(r_minus).cpu().detach().numpy())[0,1]
        corr_minus = cov3 / (stdval3 * stdval4)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("q_loss", loss1.item(), t_env)
            self.logger.log_stat("qplus_loss", loss2.item(), t_env)
            self.logger.log_stat("corr_plus", corr_plus, t_env)
            self.logger.log_stat("corr_minus", corr_minus, t_env)
            self.logger.log_stat("r_mean", r_mean.item(), t_env)
            self.logger.log_stat("r_std", r_std.item(), t_env)
            self.logger.log_stat("r_range", r_range.item(), t_env)
            self.logger.log_stat("Lmini", self.args.Lmini * Lmini.item(), t_env)
            self.logger.log_stat("Ldiv1", self.args.Ldiv1 * Ldiv1.item(), t_env)
            #self.logger.log_stat("r_in", r_in.item(), t_env)
            self.logger.log_stat("r_plus_mse", r_plus_mse.item(), t_env)
            self.logger.log_stat("r_minus_mse", r_minus_mse.item(), t_env)
            self.logger.log_stat("r_zero_loss_plus", r_plus_mse.item(), t_env)
            self.logger.log_stat("r_zero_loss_minus", r_plus_mse.item(), t_env)
            self.logger.log_stat("r_loss", r_loss.item(), t_env)
            self.logger.log_stat("acc_norewardsample_cls", acc_norewardsample_cls.item(), t_env)
            self.logger.log_stat("acc_norewardsample_ratio", acc_norewardsample_ratio.item(), t_env)
            self.logger.log_stat("celoss", celoss.item(), t_env)
            mask_elems = mask.sum().item()
            self.log_stats_t = t_env

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, show_demo=False, save_data=None):
        self.sub_train(batch, t_env, episode_num, self.mac, self.mixer, self.optimiser, self.params,
                       show_demo=show_demo, save_data=save_data)
        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        self.target_mac_plus.load_state(self.mac_plus)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.target_mixer_plus.load_state_dict(self.mixer_plus.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        self.mac_plus.cuda()
        self.target_mac_plus.cuda()
        self.sub_reward_func_pos.cuda()
        self.sub_reward_func_neg.cuda()
        self.MaskPlus.cuda()
        self.MaskMinus.cuda()
        self.reward_classify.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            self.mixer_plus.cuda()
            self.target_mixer_plus.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(th.load("{}/mixer.th".format(path),
                                                      map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    def correlation_loss(self, y_pred, y_true):
        x = y_pred.clone()
        y = y_true.clone()
        vx = x - th.mean(x)
        vy = y - th.mean(y)
        cov = th.sum(vx * vy)
        corr = cov / (th.sqrt(th.sum(vx ** 2)) * th.sqrt(th.sum(vy ** 2)) + 1e-12)
        corr = th.max(th.min(corr,th.tensor(1.0).cuda()), th.tensor(-1.0).cuda())
        return th.sub(th.tensor(1.0).cuda(), corr)