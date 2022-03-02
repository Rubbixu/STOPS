#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from ..network import *
from ..component import *
from .BaseAgent import *


class VARACNPGAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        self.task = config.task_fn()
        self.actor = config.actor_fn()
        self.criticQ = config.criticQ_fn()
        self.criticQ_opt = config.criticQ_opt_fn(self.criticQ.critic_params)
        self.criticW = config.criticW_fn()
        self.criticW_opt = config.criticW_opt_fn(self.criticW.critic_params)
        self.total_steps = 0
        self.states = self.task.reset()
        self.states = config.state_normalizer(self.states)
        self.lam = 1

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
        
            

            valueQ_loss = 0.5 * (Qreturns - self.criticQ(states)).pow(2).mean()
            valueW_loss = 0.5 * (Wreturns - self.criticW(states)).pow(2).mean()
            self.criticQ_opt.zero_grad()
            valueQ_loss.backward()
            self.criticQ_opt.step()
            
            self.criticW_opt.zero_grad()
            valueW_loss.backward()
            self.criticW_opt.step()
            
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

    
  #----------------------------------------------------------------------------------------------------------------------
  #Second Step: Train Critic
  

