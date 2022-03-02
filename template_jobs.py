from deep_rl import *
from sys import exit


def batch_mujoco():
    cf = Config()
    cf.add_argument('--i', type=int, default=0)
    cf.add_argument('--j', type=int, default=0)
    cf.merge()

    # games = [
    #     'dm-acrobot-swingup',
    #     'dm-acrobot-swingup_sparse',
    #     'dm-ball_in_cup-catch',
    #     'dm-cartpole-swingup',
    #     'dm-cartpole-swingup_sparse',
    #     'dm-cartpole-balance',
    #     'dm-cartpole-balance_sparse',
    #     'dm-cheetah-run',
    #     'dm-finger-turn_hard',
    #     'dm-finger-spin',
    #     'dm-finger-turn_easy',
    #     'dm-fish-upright',
    #     'dm-fish-swim',
    #     'dm-hopper-stand',
    #     'dm-hopper-hop',
    #     'dm-humanoid-stand',
    #     'dm-humanoid-walk',
    #     'dm-humanoid-run',
    #     'dm-manipulator-bring_ball',
    #     'dm-pendulum-swingup',
    #     'dm-point_mass-easy',
    #     'dm-reacher-easy',
    #     'dm-reacher-hard',
    #     'dm-swimmer-swimmer15',
    #     'dm-swimmer-swimmer6',
    #     'dm-walker-stand',
    #     'dm-walker-walk',
    #     'dm-walker-run',
    # ]

    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Swimmer-v2',
        'Hopper-v2',
        'Reacher-v2',
        'Ant-v2',
        'InvertedPendulum-v2',
        'InvertedDoublePendulum-v2',
    ]

    params = []

    # games = ['RiskChain-v0']
    for game in games:
        # for r in range(20, 30):
        # for r in range(10, 20):
        # for r in range(4, 10):
        for r in range(0, 4):
            # for action_noise in [0.1]:
            #     params.append([mvpi_td3_continuous, dict(game=game, run=r, lam=0, remark='mvpi_td3', EOT_eval=100, action_noise=action_noise)])
            for lam in [0.5, 1, 2]:

                params.append([mvpi_npg_continuous, dict(game=game, run=r, lam=lam, remark='mvpi_npg', EOT_eval=100)])
                # params.append([var_a2c_continuous, dict(game=game, run=r, lam=lam, remark='mva2c')])
        # for meta_prob in [0.1, 0.5, 1.0]:
            #     params.append([meta_var_ppo_continuous, dict(game=game, run=r, meta_lr=1e-3, meta_prob=meta_prob)])
            # params.append([mvp_continuous, dict(game=game, run=r, remark='mvp', EOT_eval=100)])
            # params.append([tamar_continuous, dict(game=game, run=r, remark='tamar', EOT_eval=100)])
            # params.append([risk_a2c_continuous, dict(game=game, run=r, remark='risk', EOT_eval=100)])
            # for lam in [0, 1, 2, 4, 8]:
            #     for n_samples in [10, 50, 100, 500, 1000]:
            #         params.append([off_policy_mvpi, dict(game=game, run=r, lam=lam, remark='off-policy', num_samples=n_samples)])
                # params.append([off_policy_mvpi, dict(game=game, run=r, lam=lam, use_oracle_ratio=True, remark='off-policy')])

    # for game in games:
    #     for r in range(0, 10):
    #     # for r in range(5, 10):
    #     #     params.append([ppo_continuous, dict(game=game, run=r, remark='ppo')])
    #         for meta_lr in [1e-3, 1e-2]:
    #             for meta_prob in [0.1]:
    #                 params.append([meta_var_ppo_continuous, dict(game=game, run=r, meta_lr=meta_lr, meta_prob=meta_prob)])

    # for game in games:
    #     for r in range(0, 10):
    #     # for r in range(5, 10):
    #     #     for lam in [10, 1, 0.1]:
    #         for lr in [7e-5, 7e-4]:
    #             # params.append([mvp_continuous, dict(game=game, run=r, lam=lam, remark='mvp')])
    #             # params.append([mvp_continuous, dict(game=game, run=r, lr=lr, remark='mvp')])
    #             params.append([tamar_continuous, dict(game=game, run=r, lr=lr, lam=0.1, b=10, remark='tamar')])
    #         for lam in [1, 10]:
    #             params.append([tamar_continuous, dict(game=game, run=r, lr=7e-4, lam=lam, b=10, remark='tamar')])
    #         params.append([tamar_continuous, dict(game=game, run=r, lr=7e-4, lam=0.1, b=50, remark='tamar')])

    # for game in games:
    #     for r in range(0, 10):
    #         for lr in [7e-5, 7e-4]:
    #             for lam in [0.1, 1, 10]:
    #                 params.append([risk_a2c_continuous, dict(game=game, run=r, lam=lam, lr=lr, remark='risk')])

    # params = params[0: 100]
    # params = params[100: 200]
    # params = params[200: 300]
    for d in params:
        algo, param = d
        algo(**param)

    exit()


def set_max_steps(config):
    config.max_steps = int(1e6)


def stops_ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('action_noise', 0)
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('lrC', 1e-3)
    kwargs.setdefault('lrA', 3e-4)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, action_noise=config.action_noise)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: CategoricalActorCriticNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.tanh),
        critic_body=FCBody(config.state_dim, gate=torch.tanh))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, config.lrA)
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, config.lrC)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    set_max_steps(config)
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    run_steps(STOPSPPOAgent(config))


# MVP
def mvp_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lr', 7e-4)
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('action_noise', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 16
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers, action_noise=config.action_noise)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=config.lr)
    config.network_fn = lambda: MVPNet(
        config.state_dim, config.action_dim,
        actor_body=FCBody(config.state_dim),
        critic_body=FCBody(config.state_dim))
    config.entropy_weight = 0.01
    config.gradient_clip = 0.5
    set_max_steps(config)
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    config.discount = 0.99
    run_steps(MVPAgent(config))



# MVPINPG
def mvpi_npg_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lr', 7e-4)
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('delta', 0.05)
    kwargs.setdefault('action_noise', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, action_noise=config.action_noise)
    config.eval_env = config.task_fn()

    config.actor_fn = lambda: npgActorNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.relu))
    config.critic_fn = lambda: npgCriticNet(
        config.state_dim, critic_body=FCBody(config.state_dim, gate=torch.relu))
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, config.lr)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.log_interval = 2048
    set_max_steps(config)
    config.state_normalizer = MeanStdNormalizer()
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    run_steps(MVPINPGAgent(config))

# MOPSNPG
def stops_npg_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lr', 7e-4)
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('delta', 0.05)
    kwargs.setdefault('action_noise', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, action_noise=config.action_noise)
    config.eval_env = config.task_fn()

    config.actor_fn = lambda: npgActorNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.relu))
    config.critic_fn = lambda: npgCriticNet(
        config.state_dim, critic_body=FCBody(config.state_dim, gate=torch.relu))
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, config.lr)
    config.discount = 0.99
    config.use_gae = False
    config.gae_tau = 0.95
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.log_interval = 2048
    set_max_steps(config)
    config.state_normalizer = MeanStdNormalizer()
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    run_steps(STOPSNPGAgent(config))


def varac_npg_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('action_noise', 0)
    kwargs.setdefault('delta', 0.05)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, action_noise=config.action_noise)
    config.eval_env = config.task_fn()

    config.actor_fn = lambda: npgActorNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.relu))

    config.criticQ_fn = lambda: npgCriticNet(
        config.state_dim, critic_body=FCBody(config.state_dim, gate=torch.relu))
    config.criticQ_opt_fn = lambda params: torch.optim.Adam(params, config.lr)

    config.criticW_fn = lambda: npgCriticNet(
        config.state_dim, critic_body=FCBody(config.state_dim, gate=torch.relu))
    config.criticW_opt_fn = lambda params: torch.optim.Adam(params, config.lr)

    config.lr = 1e-3
    config.discount = 0.99
    config.use_gae = False
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    set_max_steps(config)
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    run_steps(VARACNPGAgent(config))

# MOPSNPG
def mvpi_ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lrC', 1e-3)
    kwargs.setdefault('lrA', 3e-4)
    kwargs.setdefault('ppo_ratio_clip', 0.2)
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('action_noise', 0)

    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, action_noise=config.action_noise)
    config.eval_env = config.task_fn()

    config.actor_fn = lambda: npgActorNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.relu))
    config.critic_fn = lambda: npgCriticNet(
        config.state_dim, critic_body=FCBody(config.state_dim, gate=torch.relu))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, config.lrA)
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, config.lrC)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.gradient_clip = 0.5
    config.mini_batch_size = 64
    # config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    set_max_steps(config)
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    run_steps(MVPIPPOAgent(config))

# MOPSNPG
def stops_ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('lrC', 1e-3)
    kwargs.setdefault('lrA', 3e-4)
    kwargs.setdefault('ppo_ratio_clip', 0.2)
    kwargs.setdefault('lam', 1)
    kwargs.setdefault('action_noise', 0)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, action_noise=config.action_noise)
    config.eval_env = config.task_fn()

    config.actor_fn = lambda: npgActorNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.relu))
    config.critic_fn = lambda: npgCriticNet(
        config.state_dim, critic_body=FCBody(config.state_dim, gate=torch.relu))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, config.lrA)
    config.critic_opt_fn = lambda params: torch.optim.Adam(params, config.lrC)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    config.gradient_clip = 0.5
    config.mini_batch_size = 64
    # config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    set_max_steps(config)
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    run_steps(STOPSPPOAgent(config))

def varac_ppo_continuous(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('action_noise', 0)
    kwargs.setdefault('lrC', 1e-3)
    kwargs.setdefault('lrA', 3e-4)
    kwargs.setdefault('ppo_ratio_clip', 0.2)
    config = Config()
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game, action_noise=config.action_noise)
    config.eval_env = config.task_fn()

    config.actor_fn = lambda: npgActorNet(
        config.state_dim, config.action_dim, actor_body=FCBody(config.state_dim, gate=torch.relu))
    config.actor_opt_fn = lambda params: torch.optim.Adam(params, config.lrA)

    config.criticQ_fn = lambda: npgCriticNet(
        config.state_dim, critic_body=FCBody(config.state_dim, gate=torch.relu))

    config.criticQ_opt_fn = lambda params: torch.optim.Adam(params, config.lrC)
    config.criticW_fn = lambda: npgCriticNet(
        config.state_dim, critic_body=FCBody(config.state_dim, gate=torch.relu))

    config.criticW_opt_fn = lambda params: torch.optim.Adam(params, config.lrC)

    config.discount = 0.99
    config.use_gae = False
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = 10
    config.mini_batch_size = 64
    # config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    set_max_steps(config)
    config.target_kl = 0.01
    config.state_normalizer = MeanStdNormalizer()
    config.eval_interval = int(1e4)
    config.eval_episodes = 20
    run_steps(VARACPPOAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('data')
    random_seed()

    # select_device(0)
    # batch_atari()
    print(torch.cuda.is_available())
    select_device(-1)
    # batch_mujoco()

    # compute_boundary()

    # off_policy_mvpi(
    #     game='RiskChain-v0',
    #     lam=12,
    #     num_samples=int(1000),
    #     tau_lr=0.01,
    #     q_lr=0.01,
    #     pi_lr=0.01,
    # )

    # game = 'HalfCheetah-v2'
    # game = 'Walker2d-v2'
    # game = 'Ant-v2'
    # game = 'Reacher-v2'

    games = [
        'HalfCheetah-v2',
        'Walker2d-v2',
        'Swimmer-v2',
        'Hopper-v2',
        # 'Reacher-v2',
        # 'Ant-v2',
        'InvertedPendulum-v2',
        'InvertedDoublePendulum-v2',
    ]

    games = [
        'Cliffwalking-v0'
    ]
    # mvpi_td3_continuous(
    #     game=game,
    #     lam=0.1,
    #     action_noise=0.1,
    #     max_steps=int(1e4),
    #     EOT_eval=10
    # )

    # varac_continuous(
    #     game=game,
    #     lam=0.5,
    #     beta=0.01,
    #     alpha=0.01,
    #     EOT_eval=100,
    # )

    # varac_continuous(
    #     game=games[0],
    #     alpha=0.1,
    #     beta=0.1,
    #     delta=0.01,
    #     EOT_eval=100,
    # )
    n = 0
    run = 10
    # for i in [0.1, 0.5, 1, 1.5]:
    #     for j in [0.01, 0.05, 0.1, 0.5]:
    #         for k in [7e-3, 7e-4, 7e-5]:
    # for n in range(len(games)):
    # for j in [1e-1, 1e-2,1e-3,1e-4]:
    #     for k in [1e-2, 1e-3, 1e-4]:
    # for j in [0.02, 0.2, 0.5]:
    # for k in [1e-1, 1e-2, 1e-3, 1e-4]:
    #     for l in [3e-2, 3e-3, 3e-4, 3e-5]:
    #         var_ppo_continuous(
    #         game = games[n],
    #         lrC = k,
    #         lrA = l,
    #         EOT_eval=100,
    #         remark='tops_ppo'
    #     )
                # varac_npg_continuous(
                #     game=games[n],
                #     delta = j,
                #     lr = k,
                #     EOT_eval=100,
                #     remark='varac_npg'
                # )

            # mops_npg_continuous(
            #     game=games[n],
            #     delta = j,
            #     lr = k,
            #     EOT_eval=100,
            #     remark='tops_npg'
            # )
                # varac_ppo_continuous(
                #     game=games[n],
                #     ppo_ratio_clip = j,
                #     lrC = k,
                #     lrA = l,
                #     EOT_eval=100,
                #     remark='varac_ppo'
                # )

                # mops_ppo_continuous(
                #     game=games[n],
                #     ppo_ratio_clip = j,
                #     lrC = k,
                #     lrA = l,
                #     EOT_eval=100,
                #     remark='MOPS_ppo'
                # )

    lr_MOPS=[5e-4, 1e-2, 5e-3, 1e-2, 1e-2, 5e-2]
    delta_MOPS=[1e-2, 1e-2, 1e-2, 1e-2, 1e-3, 5e-3]
    lr_VARAC=[1e-4, 1e-4, 1e-2, 1e-3, 1e-3, 1e-4]
    delta_VARAC = [1e-2, 1e-2, 1e-4, 1e-2, 1e-2, 1e-3]
    # ratio_MOPS = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    # lrC_MOPS = [1e-4, 1e-3, 1e-4, 1e-3, 1e-3, 1e-3]
    # lrA_MOPS = [3e-4, 3e-4, 3e-4, 3e-4, 3e-4, 3e-4]

    ratio_VARAC = [0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    lrC_VARAC = [2.5e-4, 2.5e-3, 2.5e-2, 2.5e-3, 2.5e-2, 2.5e-2]
    lrA_VARAC = [2.5e-4, 2.5e-4, 2.5e-4, 2.5e-4, 2.5e-4, 2.5e-4]


    # for _ in range(run):
    # for n in range(len(games)):
    #     mops_npg_continuous(
    #         game=games[n],
    #         lr=lr_MOPS[n],
    #         delta=delta_MOPS[n],
    #         EOT_eval=100,
    #         remark='tops_npg'
    #     )

    # for _ in range(run):
    #     mvpi_npg_continuous(
    #         game=games[n],
    #         lr=lr_MOPS[n],
    #         delta=delta_MOPS[n],
    #         EOT_eval=100,
    #         remark='mvpi_npg'
    #     )

    # for n in range(len(games)):
    # for _ in range(run):
        # varac_npg_continuous(
        #     game=games[n],
        #     delta=delta_VARAC[n],
        #     EOT_eval=100,
        #     remark='varac_npg'
        # )

    # for _ in range(run):
    #     mvp_continuous(
    #         game=games[n],
    #         EOT_eval=100,
    #         remark='mvp'
    #     )



    # for n in range(len(games)):
    for _ in range(run):
        var_ppo_continuous(
            game = games[n],
            EOT_eval=100,
            remark='tops_ppo'
        )

    # for _ in range(run):
    #     mvpi_ppo_continuous(
    #         game = games[n],
    #         EOT_eval=100,
    #         remark='mvpi_ppo'
    #     )

    # for _ in range(run):
        # varac_ppo_continuous(
        #     game=games[n],
        #     lrA=lrA_VARAC[n],
        #     lrC=lrC_VARAC[n],
        #     ppo_ratio_clip=ratio_VARAC[n],
        #     EOT_eval=100,
        #     remark='varac_ppo'
        # )
