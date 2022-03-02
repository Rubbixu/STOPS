#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class VARACPPOAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.actor = config.actor_fn()
        self.actor_opt = config.actor_opt_fn(self.actor.actor_params)
        self.criticQ = config.criticQ_fn()
        self.criticQ_opt = config.criticQ_opt_fn(self.criticQ.critic_params)
        self.criticW = config.criticW_fn()
        self.criticW_opt = config.criticW_opt_fn(self.criticW.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.lam = 1
    
    
    
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
            vQ = self.criticQ(states)
            vW = self.criticW(states)
            next_states, rewards, terminals, info = self.task.step(to_np(prediction['a']))
            self.record_online_return(info)
            rewards = config.reward_normalizer(rewards)
            next_states = config.state_normalizer(next_states)
            storage.add(prediction)
            storage.add({'r': tensor(rewards).unsqueeze(-1),
                         'm': tensor(1 - terminals).unsqueeze(-1),
                         's': tensor(states),
                         'vQ':tensor(vQ),
                         'vW':tensor(vW)
                         })
            states = next_states
            self.total_steps += config.num_workers

        self.states = states
        prediction = self.actor(states)
        vQ = self.criticQ(states)
        vW = self.criticW(states)
        storage.add(prediction)
        storage.add({'vQ': tensor(vQ)})
        storage.add({'vW': tensor(vW)})
        storage.placeholder()

        rewards = list(storage.cat(['r']))[0]
        rewards_square = rewards ** 2
        rho_bar = rewards.mean()
        eta_bar = rewards_square.mean()
        y = rho_bar
        # self.lam = self.lam - 1/config.beta * (config.alpha + 2*y*rho_bar-eta_bar-y**2)
        

        for i in range(config.rollout_length):
            r = storage.r[i]
            storage.rsquare[i] = r**2
            storage.L[i] = (1 + 2 * self.lam * y) * r - self.lam * r**2
        # I don't need advatange in my algorithm
        advantages = tensor(np.zeros((config.num_workers, 1)))
        Qreturns = vQ.detach()
        Wreturns = vW.detach()
        returns = (1 + 2 * self.lam * y) * Qreturns - self.lam * Wreturns
        for i in reversed(range(config.rollout_length)):
            Qreturns = storage.r[i] + config.discount * storage.m[i] * Qreturns
            Vreturns = storage.rsquare[i] + config.discount * storage.m[i] * Wreturns
            returns = storage.L[i] + config.discount * storage.m[i] * returns
            Li = (1 + 2 * self.lam * y) * storage.vQ[i] - self.lam * storage.vW[i]
            Li1 = (1 + 2 * self.lam * y) * storage.vQ[i+1] - self.lam * storage.vW[i+1]
            if not config.use_gae:
                advantages = returns - Li.detach()
            else:
                td_error = storage.L[i] + config.discount * storage.m[i] * Li1 - Li
                advantages = advantages * config.gae_tau * config.discount * storage.m[i] + td_error
            
            storage.adv[i] = advantages.detach()
            storage.Qret[i] = Qreturns.detach()
            storage.Wret[i] = Wreturns.detach()
            
        states, actions, log_probs_old, Qreturns, Wreturns, advantages, rewards = storage.cat(['s', 'a', 'log_pi_a', 'Qret', 'Wret','adv', 'r'])
        actions = actions.detach()
        log_probs_old = log_probs_old.detach()
        advantages = (advantages - advantages.mean()) / advantages.std()

        for _ in range(config.optimization_epochs):
            sampler = random_sample(np.arange(states.size(0)), config.mini_batch_size)
            for batch_indices in sampler:
                batch_indices = tensor(batch_indices).long()
                sampled_states = states[batch_indices]
                sampled_actions = actions[batch_indices]
                sampled_log_probs_old = log_probs_old[batch_indices]
                sampled_Qreturns = Qreturns[batch_indices]
                sampled_Wreturns = Wreturns[batch_indices]
                sampled_advantages = advantages[batch_indices]

                prediction = self.actor(sampled_states, sampled_actions)
                ratio = (prediction['log_pi_a'] - sampled_log_probs_old).exp()
                obj = ratio * sampled_advantages
                obj_clipped = ratio.clamp(1.0 - self.config.ppo_ratio_clip,
                                          1.0 + self.config.ppo_ratio_clip) * sampled_advantages
                policy_loss = -torch.min(obj, obj_clipped).mean() - config.entropy_weight * prediction['ent'].mean()
    
                valueQ_loss = 0.5 * (sampled_Qreturns - self.criticQ(sampled_states)).pow(2).mean()
                valueW_loss = 0.5 * (sampled_Wreturns - self.criticW(sampled_states)).pow(2).mean()

                approx_kl = (sampled_log_probs_old - prediction['log_pi_a']).mean()
                if approx_kl <= 1.5 * config.target_kl:
                    self.actor_opt.zero_grad()
                    policy_loss.backward()
                    self.actor_opt.step()
            
            self.criticQ_opt.zero_grad()
            valueQ_loss.backward()
            self.criticQ_opt.step()
            
            self.criticW_opt.zero_grad()
            valueW_loss.backward()
            self.criticW_opt.step()
        
            





    
  #----------------------------------------------------------------------------------------------------------------------
  #Second Step: Train Critic
  

