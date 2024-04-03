import numpy as np
import torch
import envs, neihbourhoods, drifts
from agents import *
import torch.random
//# 设置随机种子
np.random.seed(0)
torch.manual_seed(0)

class Training:

    def __init__(self, config):
        //# 从配置中获取环境、漂移函数、邻域和采样方法
        envs = config["envs"]
        drifts = config["drifts"]
        neighbourhoods = config["neighbourhoods"]
        samplings = config["samplings"]

        self.n_threads = len(envs)# 线程数量
        self.envs = envs
        # 初始化代理列表，每个代理对应一个环境、漂移函数、邻域和采样方法
        self.agents = [Agent(envs[i], neighbourhoods[i], drifts[i], samplings[i])
                        for i in range(self.n_threads)]
# 计算平均价值函数
    def get_mean_v(self, env, agent):
        # 计算适应性价值函数
        fitted_v = env.fit_v(soft_to_direct(agent.pi.detach()))
        mean_v, n_states = 0., 0
        # 对每个状态计算平均价值
        for state in env.states:
            if not hasattr(env, "barrier") or state not in env.barrier:
                mean_v += fitted_v[state]
                n_states += 1
        mean_v /= n_states
        return mean_v
 
    # 训练方法
    def train(self, n_iters=20):

        min_drift_val = np.zeros((n_iters+1, self.n_threads))# 最小漂移值
        exp_drift_val = np.zeros((n_iters+1, self.n_threads))# 期望漂移值
        V = np.zeros((n_iters+1, self.n_threads))# 价值函数估计值
        # 计算初始价值函数估计值
        V[0, :] = np.array([
            self.get_mean_v(self.envs[th], self.agents[th]) for th in range(self.n_threads)
            ])

        for i in range(n_iters):

            # 计算每个线程中环境的动作值函数
            Q = [self.envs[th].fit_q(soft_to_direct(self.agents[th].pi.detach()))
                for th in range(self.n_threads) ]
            # 更新每个代理的策略并获取最小漂移值和期望漂移值
            _ = list(zip(*[self.agents[th].mirror_step(Q[th]) for th in range(self.n_threads)]))

            min_drift, exp_drift = _[0], _[1]
            min_drift_val[i+1, :] = min_drift_val[i, :] + np.array(min_drift)
            exp_drift_val[i+1, :] = exp_drift_val[i, :] + np.array(exp_drift)

            # 计算每个线程中环境的价值函数估计值
            V[i+1, :] = np.array([
                self.envs[th].fit_v(soft_to_direct(self.agents[th].pi.detach())).mean().item()
                for th in range(self.n_threads) ])

        return {"V": V, "min_drift": min_drift_val, "exp_drift": exp_drift_val}

