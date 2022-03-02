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


class MVPINPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.actor = config.actor_fn()
        self.critic = config.critic_fn()
        self.critic_opt = config.critic_opt_fn(self.critic.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        # self.replay = config.replay_fn()
        # self.random_process = config.random_process_fn()
        # self.state = None
        # self.online_rewards = deque(maxlen=int(1e4))
        
    
    def flat_grad(self, grads):
        grad_flatten = []
        for grad in grads:
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten)
        return grad_flatten
    
    def flat_params(self):
        params = []
        for param in self.actor.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params)
        return params_flatten
    
    
    def kl_divergence(self, states):
        
        prediction_old = self.actor(states)
        mu_old = prediction_old['mean'].detach()
        std_old = prediction_old['std'].detach()
        
        prediction = self.actor(states)
        mu = prediction['mean']
        std = prediction['std']

    
        # kl divergence between old policy and new policy : D( pi_old || pi_new )
        # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
        # be careful of calculating KL-divergence. It is not symmetric metric
        kl = torch.log(std/std_old) + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
             (2.0 * std.pow(2)) - 0.5
        # return kl.sum(-1).unsqueeze(-1)
        return kl.sum(-1, keepdim=True).mean()
    
    def flat_hessian(self, hessians):
        hessians_flatten = []
        for hessian in hessians:
            hessians_flatten.append(hessian.contiguous().view(-1))
        hessians_flatten = torch.cat(hessians_flatten).data
        return hessians_flatten
    
    def fisher_vector_product(self, states, p):
        p.detach()
        kl = self.kl_divergence(states)
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad = self.flat_grad(kl_grad)  # check kl_grad == 0
    
        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = torch.autograd.grad(kl_grad_p, self.actor.parameters())
        kl_hessian_p = self.flat_hessian(kl_hessian_p)
        return kl_hessian_p + 0.1 * p
    
    def conjugate_gradient(self, states, b, nsteps, EPS=1e-8, residual_tol=1e-10):
        x = torch.zeros(b.size())
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)
        for i in range(nsteps):
            _Avp = self.fisher_vector_product(states, p)
            alpha = rdotr / (torch.dot(p, _Avp)+EPS)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break

        return x    
    
    def update_model(self, new_params):
        index = 0
        for params in self.actor.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length
        
    
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
    
                v = self.critic(states)
                value_loss = F.mse_loss(v, returns)
                self.critic_opt.zero_grad()
                value_loss.backward()
                self.critic_opt.step()
            #----------------------------------------------------------------------------------------------------------------------
            #Third Step: Get gradient of loss and hessian of kl
            prediction = self.actor(states, actions)
            ratio = (prediction['log_pi_a'] - log_probs_old).exp()
            # obj = prediction['log_pi_a'] * (returns.unsqueeze(1))
            obj = ratio * advantages
            obj = obj.mean()
            
            loss_grad = torch.autograd.grad(obj, self.actor.parameters())
    
            # print(loss_grad.size())
            loss_grad = self.flat_grad(loss_grad)
            step_dir = self.conjugate_gradient(states, loss_grad.data, nsteps=10)
            
                # ----------------------------
            # step 4: get step direction and step size and update actor
            gHg = (self.fisher_vector_product(states, step_dir) * step_dir).sum(0)
            step_size = torch.sqrt(2 * config.delta / gHg)
            params = self.flat_params()
            new_params = params + step_size * step_dir
            self.update_model(new_params)
            
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
            
            
        
        



