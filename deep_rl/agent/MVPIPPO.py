#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *
import torchvision
from collections import deque


class MVPIPPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.actor = config.actor_fn()
        self.actor_opt = config.actor_opt_fn(self.actor.actor_params)
        self.critic = config.critic_fn()
        self.critic_opt = config.critic_opt_fn(self.critic.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        # self.replay = config.replay_fn()
        # self.random_process = config.random_process_fn()
        # self.state = None
        # self.online_rewards = deque(maxlen=int(1e4))
        
          
    
    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        prediction = self.actor(state)
        self.config.state_normalizer.unset_read_only()
        return to_np(prediction['a'])

    def step(self):
        config = self.config
        storage = Storage(config.rollout_length) #Store batch data
        states = self.states #It is plurral because multiple workers simultaneously
        # Collecting transition
        for _ in range(config.rollout_length):
            prediction = self.actor(states)
            v = self.critic(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r_original': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states),
                         'v': tensor(v)
                         })
            states = next_states
            self.total_steps += config.num_workers*config.optimization_epochs

        self.states = states
        prediction = self.actor(states)
        v = self.critic(states)
        storage.add(prediction)
        storage.add({'v': tensor(v)})
        storage.placeholder()

        rewards = list(storage.cat(['r_original']))[0]
        y = rewards.mean()

        for i in range(config.rollout_length):
            r_original = storage.r_original[i]
            storage.r[i] = r_original - config.lam * r_original ** 2 + 2 * config.lam * r_original * y

        advantages = tensor(np.zeros((config.num_workers, 1)))
        returns = v.detach()
        #----------------------------------------------------------------------------------------------------------------------
        #First Step: Get Return
        for i in reversed(range(config.rollout_length)):
            returns = storage.r[i] + config.discount * storage.m[i] * returns
            if not config.use_gae:
                advantages = returns - storage.v[i].detach()
            else:
                td_error = storage.r[i] + config.discount * storage.m[i] * storage.v[i + 1] - storage.v[i]
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            storage.adv[i] = advantages.detach()
            storage.ret[i] = returns.detach()
        # returns = (returns - returns.mean()) / returns.std()
        states, actions, log_probs_old, returns, advantages, rewards = storage.cat(['s', 'a', 'log_pi_a', 'ret', 'adv', 'r_original'])
        # states, actions, returns, rewards = storage.cat(['s', 'a', 'ret', 'r_original'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()
        #----------------------------------------------------------------------------------------------------------------------
        #Second Step: Train Critic
        
        
        for _ in range(config.optimization_epochs):
            for _ in range(config.optimization_epochs):
                sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
                for batch_indices in sampler:
                    batch_indices = tensor(batch_indices).long()
                    sampled_states = states[batch_indices]
                    sampled_actions = actions[batch_indices]
                    sampled_log_probs_old = log_probs_old[batch_indices]
                    sampled_returns = returns[batch_indices]
                    sampled_advantages = advantages[batch_indices]
    
                    prediction = self.actor(sampled_states, sampled_actions)
                    ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                    obj = ratio * sampled_advantages
                    obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                              1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                    policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()
        
                    value_loss = 0.5 * (sampled_returns - self.critic(sampled_states)).pow(2).mean()
    
                    approx_kl = (sampled_log_probs_old - prediction['log_pi_a']).mean()
                    if approx_kl <= 1.5 * config.target_kl:
                        self.actor_opt.zero_grad()
                        policy_loss.backward()
                        self.actor_opt.step()
    
                    self.critic_opt.zero_grad()
                    value_loss.backward()
                    self.critic_opt.step()
            
        # def step(self):
        #     config = self.config
        #     if self.state is None:
        #         self.random_process.reset_states()
        #         # How is it that self.network take the same input as in VarPPO, but return a different
        #         # type of result. (Action, vs. Prediction['a'])
        #         # Because they have different network
        #         self.state = self.task.reset()
        #         self.state = config.state_normalizer(self.state)
    
        #     if self.total_steps < config.warm_up:
        #         action = [self.task.action_space.sample()]
        #     else:
        #         prediction = self.actor(self.state)
        #         action = to_np(prediction['a'])
        #     next_state, reward, done, info = self.task.step(action)
        #     next_state = self.config.state_normalizer(next_state)
        #     self.record_online_return(info)
        #     reward = self.config.reward_normalizer(reward)
        #     self.online_rewards.append(reward[0])
    
        #     experiences = list(zip(self.state, action, reward, next_state, done))
        #     self.replay.feed_batch(experiences)
        #     if done[0]:
        #         self.random_process.reset_states()
        #     self.state = next_state
        #     self.total_steps += 1
    
        #     if self.replay.size() >= config.warm_up:
        #         y = np.mean(self.online_rewards)
        #         experiences = self.replay.sample()
        #         states, actions, rewards, next_states, terminals = experiences
        #         states = tensor(states)
        #         actions = tensor(actions)
        #         rewards = tensor(rewards).unsqueeze(-1)
        #         rewards = rewards - config.lam * rewards.pow(2) + 2 * config.lam * rewards * y
        #         next_states = tensor(next_states)
        #         mask = tensor(1 - terminals).unsqueeze(-1)
    
        #         for _ in range(config.optimization_epochs):
        #             v = self.critic(states)
        #             value_loss = F.mse_loss(v, rewards)
        #             self.critic_opt.zero_grad()
        #             value_loss.backward()
        #             self.critic_opt.step()
            
            
        
        



