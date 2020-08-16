#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`samplesizelib.linear.models` contains classes:
- :class:`samplesizelib.linear.models.LinearModel`
- :class:`samplesizelib.linear.models.RegressionModel`
- :class:`samplesizelib.linear.models.LogisticModel`
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

import numpy as np
import scipy.stats as sps
from scipy.special import expit as expit

from sklearn.linear_model import LogisticRegression as LogisticRegression


class LinearModel(object):
    def __init__(self, y, X, **kwards):
        pass
    
    def fit(self):
        NotImplementedError
    
    def predict(self, params, X=None):
        NotImplementedError
    
    def loglike(self, params):
        NotImplementedError

    def score(self, params):
        NotImplementedError

    def hessian(self, params):
        NotImplementedError

    def loglike_fixed(self, params):
        NotImplementedError

    def score_fixed(self, params):
        NotImplementedError

    def hessian_fixed(self, params):
        NotImplementedError

    def covariance(self, params):
        NotImplementedError

class RegressionModel(LinearModel):
    def __init__(self, y, X, **kwards):
        self.y = y
        self.X = X
        self.alpha = kwards.pop('alpha', 0.01)
        self.w = None

        self.n = self.X.shape[1]
        self.m = self.y.shape[0]

        self.prior = sps.multivariate_normal(
            mean = np.zeros(self.X.shape[1]), 
            cov = self.alpha*np.eye(self.X.shape[1]))

        self.log2pi = np.log(2*np.pi)

    def fit(self):
        self.w = np.linalg.inv(self.alpha*np.eye(self.n) + self.X.T@self.X)@self.X.T@self.y
        return self.w

    def predict(self, params, X=None):
        if X is None:
            X = self.X
        return X@params

    def loglike(self, params):
        return 0.5*(-np.sum((self.y - self.X@params)**2) - self.m*self.log2pi)

    def score(self, params):
        return self.X.T@self.y - self.X.T@self.X@params

    def hessian(self, params):
        return -self.X.T@self.X

    def loglike_fixed(self, params):
        return self.loglike(params) + self.prior.logpdf(params)

    def score_fixed(self, params):
        return self.score(params) - self.alpha*params

    def hessian_fixed(self, params):
        return self.hessian(params) - self.alpha*np.eye(self.n)
    
    def covariance(self, params):
        return np.linalg.inv(-self.hessian_fixed(params))


class LogisticModel(LinearModel):
    def __init__(self, y, X, **kwards):
        self.y = y
        self.X = X
        self.alpha = kwards.pop('alpha', 0.01)
        self.w = None

        self.n = X.shape[1]
        self.m = y.shape[0]

        self.prior = sps.multivariate_normal(
            mean = np.zeros(self.X.shape[1]), 
            cov = self.alpha*np.eye(self.X.shape[1]))

    def fit(self):
        model_sk_learn = LogisticRegression(C = 1./self.alpha)
        model_sk_learn.fit(self.X, self.y)
        self.w = model_sk_learn.coef_[0]
        return self.w

    def predict(self, params, X = None):
        if X is None:
            X = self.X
        return expit(X@params)

    def loglike(self, params):
        epsilon = 10**(-10)
        q = 2*self.y - 1
        res = expit(q*np.dot(self.X, params))
        res = res + (res < epsilon)*epsilon
        return np.sum(np.log(res))

    def score(self, params):
        theta = expit(self.X@params)
        return np.dot(self.y - theta, self.X)

    def hessian(self, params):
        theta = expit(self.X@params)
        return -np.dot(theta*(1-theta)*self.X.T, self.X)

    def loglike_fixed(self, params):
        return self.loglike(params) + self.prior.logpdf(params)

    def score_fixed(self, params):
        return self.score(params) - self.alpha*params

    def hessian_fixed(self, params):
        return self.hessian(params) - self.alpha*np.eye(self.n)

    def covariance(self, params):
        return np.linalg.inv(-self.hessian_fixed(params))
