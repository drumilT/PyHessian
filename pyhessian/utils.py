#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np
from math import ceil

def percentile(tensor, p):
    """
    Returns percentile of tensor elements

    Arguments:
        tensor {torch.Tensor} -- a tensor to compute percentile
        p {float} -- percentile (values in [0,1])
    """
    if p > 1.:
        raise ValueError(f'Percentile parameter p expected to be in [0, 1], found {p:.5f}')
    k = ceil(tensor.numel() * (1 - p))
    if p == 0:
        return -1 # by convention all param_stats >= 0
    return torch.topk(tensor.view(-1), k)[0][-1]

def get_weight_mask(param_stats, drop):
    """
    Returns mask for param_stats such that the top drop% are not included
    Arguments:
        param_stats {torch.Tensor} -- a tensor to compute w.r.t to param stats
        drop {float} -- percentile (values in [0,1])
    """    
    if param_stats is None: return None
    threshold = percentile(param_stats, drop)
    return (param_stats < threshold).float()

def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return [torch.sum(x * y) for (x, y) in zip(xs, ys)]


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'bias' in name:
            continue
        names.append(name)
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)

    print("Names of Params",names)
    print("Number of Params", len(names))

    return names, params, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH,
                             params,
                             grad_outputs=v,
                             only_inputs=True,
                             retain_graph=True)
    return hv


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)
