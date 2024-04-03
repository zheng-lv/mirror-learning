import numpy as np
import torch

def Zero():
    return lambda pi1, pi2: torch.zeros(pi1.shape[0])
//Zero() 函数返回一个用于计算零漂移的函数，它将输入的两个策略都视为参数，并返回一个形状与第一个策略相同的零张量。
def TVSq(scale=1.):
    return lambda pi1, pi2: scale*0.5*torch.sum(torch.abs(pi1-pi2), dim=-1)**2
//TVSq(scale=1.) 函数返回一个用于计算总变差平方的函数。总变差平方（Total Variation Squared）是两个策略之间的差异的一种度量。
def EuclidSq(scale=1.):
    return lambda pi1, pi2: scale*torch.sum( (pi1-pi2)**2, dim=-1 )
//EuclidSq(scale=1.) 函数返回一个用于计算欧几里得平方的函数。欧几里得平方是两个策略之间欧几里得距离的平方。
def KL(scale=1.):
    return lambda pi1, pi2: scale*torch.sum(pi1*torch.log((pi1+1e-6)/(pi2+1e-6)), dim=-1)
//KL(scale=1.) 函数返回一个用于计算KL散度的函数。KL散度是两个策略之间的相对熵

//代码还定义了两个用于计算期望漂移的函数：
def expected_drift(mirror, beta, functional=True):
    if functional:
        return lambda Pi1, Pi2: torch.dot(beta(Pi1), mirror(Pi1, Pi2))

    return lambda Pi1, Pi2: torch.dot(beta, mirror(Pi1, Pi2))
//expected_drift(mirror, beta, functional=True) 函数返回一个用于计算期望漂移的函数。它采用两个策略的分布 Pi1 和 Pi2 作为输入，并根据给定的镜像函数 mirror 和权重向量 beta 计算期望漂移。

def min_drift(drift):
    return lambda Pi1, Pi2: torch.min(drift(Pi1, Pi2))
//min_drift(drift) 函数返回一个用于计算最小漂移的函数。它采用两个策略的分布 Pi1 和 Pi2 作为输入，并根据给定的漂移函数 drift 计算最小漂移。
