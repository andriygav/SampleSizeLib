#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`samplesizelib.linear.statistical` contains classes:
- :class:`samplesizelib.linear.statistical.LagrangeEstimator`
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

import numpy as np
import scipy.stats as sps
from scipy.optimize import minimize as minimize

from ..shared.estimator import SampleSizeEstimator
from .models import RegressionModel, LogisticModel


class LagrangeEstimator(SampleSizeEstimator):
    r"""
    Description of Lagrange Method

    :param statmodel: the machine learning algorithm
    :type statmodel: RegressionModel or LogisticModel
    :param ind_u: to do
    :type ind_u: numpy.ndarray
    :param epsilon: to do
    :type epsilon: float
    :param alpha: to do
    :type alpha: float
    :param beta: to do
    :type beta: float
    """

    def __init__(self, statmodel, **kwards):
        r"""Constructor method
        """
        self.statmodel = statmodel

        self.ind_u = kwards.pop('ind_u', None)
        if not isinstance(self.ind_u, np.ndarray) and self.ind_u:
            raise ValueError(
                "The ind_u should be numpy.ndarray but get {}".format(
                    self.ind_u))

        self.epsilon = kwards.pop('epsilon', 0.3)
        if self.epsilon <= 0:
            raise ValueError(
                "The epsilon must be positive value but get {}".format(
                    self.epsilon))
        self.alpha = kwards.pop('alpha', 0.05)
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError(
                "The alpha must be between 0 and 1 but get {}".format(
                    self.alpha))
        self.beta = kwards.pop('beta', 0.05)
        if self.beta < 0 or self.alpha > 1:
            raise ValueError(
                "The beta must be between 0 and 1 but get {}".format(
                    self.beta))

    def _fix_variables(self, f, x1, ind_1, dim = 0):
        r"""
        Return ...
        """
        ind_2 = (ind_1 == False)
        if dim == 0:
            return lambda x2: f(self._stitch_vectors(x1, x2, ind_1))
        elif dim == 1: 
            return lambda x2: f(self._stitch_vectors(x1, x2, ind_1))[ind_2]
        elif dim == 2:
            return lambda x2: f(self._stitch_vectors(x1, x2, ind_1))[ind_2][:,ind_2]
        else:
            raise ValueError(
                'dim must be between 0 and 2 but get {}'.format(dim))

    @staticmethod
    def _negative_func(f):
        r"""
        Return ...
        """
        negative_func_fx = lambda x, *args: -f(x, *args)
        negative_func_f = lambda x, *args: negative_func_fx(x, *args)
        return negative_func_f

    @staticmethod
    def _stitch_vectors(x1, x2, ind_1):
        r"""
        Return ...
        """
        x = np.zeros(ind_1.size)
        x[ind_1] = x1
        x[ind_1 == False] = x2
        return x

    @staticmethod
    def _get_gamma(ind_u, alpha, beta):
        r"""
        Return ...
        """
        k = ind_u.sum()
        f = lambda x: np.abs(sps.chi2(k, loc=x).ppf(beta) - sps.chi2(k).ppf(1-alpha))
        gamma = minimize(f, 0.)['x'][0]
        return gamma

    def forward(self, features, target):
        r"""
        Returns sample size prediction for the given dataset.
        
        :param features: The tensor of shape
            `num_elements` :math:`\times` `num_feature`.
        :type features: array.
        :param target: The tensor of shape `num_elements`.
        :type target: array.
        
        :return: sample size estimation for the given dataset.
        :rtype: dict
        """
        y, X = target, features
        m, n = features.shape
        if self.ind_u is None:
            ind_u = np.concatenate([np.ones(n // 2), np.zeros(n - n//2)]).astype(bool)
        else:
            ind_u = self.ind_u

        ind_v = ind_u == False

        model = self.statmodel(y, X)

        w_hat = model.fit()

        mu = model.predict(w_hat)


        if len(list(set(list(y)))) == 2:
            v = mu*(1-mu)
        else:
            v = np.ones_like(y)*(mu-y).var()


        wu0 = w_hat[ind_u] + self.epsilon

        wv_hat = minimize(self._fix_variables(self._negative_func(model.loglike_fixed), wu0, ind_u), np.zeros(ind_v.sum()),
                     jac = self._fix_variables(self._negative_func(model.score_fixed), wu0, ind_u, 1),
                     hess = self._fix_variables(self._negative_func(model.hessian_fixed), wu0, ind_u, 2),
                     method = 'Newton-CG')['x']

        w_0 = self._stitch_vectors(wu0, wv_hat, ind_u)

        I = -model.hessian_fixed(w_0)
        I_muv = I[ind_u][:,ind_v]
        I_mvv = I[ind_v][:,ind_v]
        
        Z_star = (X[:,ind_u].T - I_muv @ np.linalg.inv(I_mvv) @ X[:,ind_v].T).T
        Z_star_matrices = np.asarray([Z_star[i,None].T @ Z_star[i, None] for i in range(m)])
        
        delta = np.ones_like(y)
        mu_star = model.predict(w_0)
        
        xi_m = (((mu - mu_star)*delta[None,:]).T * Z_star).sum(0)
        Sigma_m = ((v * delta**2).reshape(-1,1,1) * Z_star_matrices).sum(0)
        
        gamma_0 = (xi_m @ np.linalg.inv(Sigma_m) @ xi_m)/m
        gamma = self._get_gamma(ind_u, self.alpha, self.beta)
        
        m_star = np.ceil(gamma/gamma_0).astype(int)
        return {'m*': m_star}

