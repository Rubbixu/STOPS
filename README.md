The code for the paper

*STOPS: Short-term Volatility-controlled Policy Search and its Global Convergence* \
Liangliang Xu, Daoming Lyu, Yangchen Pan, Aiwen Jiang, Bo Liu

The code is modified from
https://github.com/ShangtongZhang/DeepRL/tree/MVPI
    .              
    ├── requirements.txt                                # Dependencies
    ├── template_jobs.py                                # Entrance for the experiments
    |   ├── stops_npg_continuous                        # STOPS-NPG calling
    |   ├── stops_ppo_continuous                        # STOPS-PPO calling
    |   ├── varac_npg_continuous                        # VARAC-NPG calling
    |   ├── varac_ppo_continuous                        # VARAC-PPO calling
    |   ├── mvpi_npg_continuous                         # MVPI-NPG calling
    |   ├── mvpi_ppo_continuous                         # MVPI-PPO calling
    |   ├── mvp_continuous                              # MVP calling
    ├── deep_rl/agent/STOPSNPG.py                       # STOPS-NPG implementation
    ├── deep_rl/agent/STOPSPPO.py                       # STOPS-PPO implementation
    ├── deep_rl/agent/VARACNPG.py                       # VARAC-NPG implementation
    ├── deep_rl/agent/VARACPPO.py                       # VARAC-PPO implementation
    ├── deep_rl/agent/MVPINPG.py                        # MVPI-NPG implementation
    ├── deep_rl/agent/MVPIPPO.py                        # MVPI-PPO implementation
    └── log/plot.py                                     # Plotting

> Algorithm implementations not used in the paper may be broken and should never be used.
