#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`samplesizelib.linear.bayesian` contains classes:
- :class:`samplesizelib.linear.bayesian.APVCEstimator`
- :class:`samplesizelib.linear.bayesian.ACCEstimator`
- :class:`samplesizelib.linear.bayesian.ALCEstimator`
- :class:`samplesizelib.linear.bayesian.MaxUtilityEstimator`
- :class:`samplesizelib.linear.bayesian.KLEstimator`
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

from multiprocessing import Pool

import numpy as np
from tqdm import tqdm
import scipy.stats as sps
from scipy.optimize import minimize_scalar

from ..shared.estimator import SampleSizeEstimator
from ..shared.utils import Dataset

class APVCEstimator(SampleSizeEstimator):
    r"""
    Description of APVC Method

    :param statmodel: the machine learning algorithm
    :type statmodel: RegressionModel or LogisticModel
    :param averaging: to do
    :type averaging: float
    :param epsilon: to do
    :type epsilon: float
    :param begin: to do
    :type begin: int
    :param end: to do
    :type end: int
    :param num: to do
    :type num: int
    :param multiprocess: to do
    :type multiprocess: bool
    :param progressbar: to do
    :type progressbar: bool
    """

    def __init__(self, statmodel, **kwards):
        r"""Constructor method
        """
        self.statmodel = statmodel

        self.averaging = int(kwards.pop('averaging', 100))
        if self.averaging <= 0:
            raise ValueError(
                "The averaging should be positive but get {}".format(
                    self.averaging))

        self.epsilon = kwards.pop('epsilon', 0.5)
        if self.epsilon <= 0:
            raise ValueError(
                "The epsilon must be positive value but get {}".format(
                    self.epsilon))
        
        self.begin = kwards.pop('begin', None)
        if self.begin is not None and self.begin < 0:
            raise ValueError(
                "The begin must be positive value but get {}".format(
                    self.begin))

        self.end = kwards.pop('end', None)
        if self.end is not None and self.end < 0:
            raise ValueError(
                "The end must be positive value but get {}".format(
                    self.end))

        if self.end is not None and self.begin is not None and self.end <= self.begin:
            raise ValueError(
                "The end value must be greater than the begin value but {}<={}".format(
                    self.end, self.begin))

        self.num = kwards.pop('num', 5)
        if self.num <=0:
            raise ValueError(
                "The num must be positive value but get {}".format(
                    self.num))
        if self.end is not None and self.begin is not None and self.num >= self.end - self.begin:
            raise ValueError(
                "The num value must be smaler than (end - begin) but {}>={}".format(
                    self.num, self.end - self.begin))

        self.multiprocess = kwards.pop('multiprocess', False)
        if not isinstance(self.multiprocess, bool):
            raise ValueError(
                "The multiprocess must be bool value but get {}".format(
                    self.multiprocess))

        self.progressbar = kwards.pop('progressbar', False)
        if not isinstance(self.progressbar, bool):
            raise ValueError(
                "The progressbar must be bool value but get {}".format(
                    self.progressbar))

        if kwards:
            raise ValueError("Invalid parameters: %s" % str(kwards))

        self.dataset = None

    def _hDispersion(self, dataset):
        r"""
        Return ...
        """
        X, y = dataset.sample()

        w_hat = self.statmodel(y, X).fit()

        cov = np.linalg.inv(
        	0.01*np.eye(w_hat.shape[0]) - self.statmodel(y, X).hessian(w_hat))
        return np.sqrt(np.sum((np.linalg.eigvals(cov)/2)**2))

    def _score_subsample(self, m):
        r"""
        Return ...
        """
        X_m, y_m = self.dataset.sample(m)
        dataset_m = Dataset(X_m, y_m)
        return self._hDispersion(dataset_m)

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

        self.dataset = Dataset(features, target)

        if self.end is None:
            end = len(self.dataset) - 1
        else:
            end = self.end

        if self.begin is None:
            begin = 2*self.dataset.n
        else:
            begin = self.begin

        if end <= begin:
            raise ValueError(
                "The end value must be greater than the begin value but {}<={}".format(
                    end, begin))

        if self.num >= end - begin:
            raise ValueError(
                "The num value must be smaler than (end - begin) but {}>={}".format(
                    self.num, end - begin))

        subset_sizes = np.arange(begin, end, self.num, dtype=np.int64)

        list_of_answers = []
        points_one = np.ones(self.averaging, dtype=np.int64)

        if self.multiprocess:
            pool = Pool()
            mapping = pool.map
        else:
            mapping = map

        if self.progressbar:
            iterator = tqdm(subset_sizes)
        else:
            iterator = subset_sizes

        for m in iterator:
            list_of_answers.append(
                np.asarray(
                    list(mapping(self._score_subsample, m*points_one))))

        if self.multiprocess:
            pool.close()
            pool.join()

        list_of_answers = np.asarray(list_of_answers)

        list_of_E = np.mean(list_of_answers, axis = 1)
        list_of_S = np.std(list_of_answers, axis = 1)

        m_size = end
        for m, mean in zip(reversed(subset_sizes), reversed(list_of_E)):
            if mean < self.epsilon:
                m_size = m

        return {'m*': m_size,
                'E': np.array(list_of_E),
                'S': np.array(list_of_S),
                'm': np.array(subset_sizes),
               }

class ACCEstimator(SampleSizeEstimator):
    r"""
    Description of ACC Method

    :param statmodel: the machine learning algorithm
    :type statmodel: RegressionModel or LogisticModel
    :param averaging: to do
    :type averaging: float
    :param alpha: to do
    :type alpha: float
    :param length: to do
    :type length: float
    :param begin: to do
    :type begin: int
    :param end: to do
    :type end: int
    :param num: to do
    :type num: int
    :param multiprocess: to do
    :type multiprocess: bool
    :param progressbar: to do
    :type progressbar: bool
    """

    def __init__(self, statmodel, **kwards):
        r"""Constructor method
        """
        self.statmodel = statmodel

        self.averaging = int(kwards.pop('averaging', 100))
        if self.averaging <= 0:
            raise ValueError(
                "The averaging should be positive but get {}".format(
                    self.averaging))

        self.length = kwards.pop('length', 0.25)
        if self.length <= 0:
            raise ValueError(
                "The length must be positive value but get {}".format(
                    self.length))

        self.alpha = kwards.pop('alpha', 0.05)
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError(
                "The alpha must be between 0 and 1 but get {}".format(
                    self.alpha))
        
        self.begin = kwards.pop('begin', None)
        if self.begin is not None and self.begin < 0:
            raise ValueError(
                "The begin must be positive value but get {}".format(
                    self.begin))

        self.end = kwards.pop('end', None)
        if self.end is not None and self.end < 0:
            raise ValueError(
                "The end must be positive value but get {}".format(
                    self.end))

        if self.end is not None and self.begin is not None and self.end <= self.begin:
            raise ValueError(
                "The end value must be greater than the begin value but {}<={}".format(
                    self.end, self.begin))

        self.num = kwards.pop('num', 5)
        if self.num <=0:
            raise ValueError(
                "The num must be positive value but get {}".format(
                    self.num))
        if self.end is not None and self.begin is not None and self.num >= self.end - self.begin:
            raise ValueError(
                "The num value must be smaler than (end - begin) but {}>={}".format(
                    self.num, self.end - self.begin))

        self.multiprocess = kwards.pop('multiprocess', False)
        if not isinstance(self.multiprocess, bool):
            raise ValueError(
                "The multiprocess must be bool value but get {}".format(
                    self.multiprocess))

        self.progressbar = kwards.pop('progressbar', False)
        if not isinstance(self.progressbar, bool):
            raise ValueError(
                "The progressbar must be bool value but get {}".format(
                    self.progressbar))

        if kwards:
            raise ValueError("Invalid parameters: %s" % str(kwards))

        self.dataset = None

    def _iDistribution(self, dataset):
        r"""
        Return ...
        """
        X, y = dataset.sample()

        w_hat = self.statmodel(y, X).fit()

        cov = np.linalg.inv(
            0.01*np.eye(w_hat.shape[0]) - self.statmodel(y, X).hessian(w_hat))
        
        W = sps.multivariate_normal(mean=np.zeros(w_hat.shape[0]), cov = cov).rvs(size=1000)

        return (np.sqrt((W**2).sum(axis=1)) < 3*self.length).mean()

    def _score_subsample(self, m):
        r"""
        Return ...
        """
        X_m, y_m = self.dataset.sample(m)
        dataset_m = Dataset(X_m, y_m)
        return self._iDistribution(dataset_m)

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

        self.dataset = Dataset(features, target)

        if self.end is None:
            end = len(self.dataset) - 1
        else:
            end = self.end

        if self.begin is None:
            begin = 2*self.dataset.n
        else:
            begin = self.begin

        if end <= begin:
            raise ValueError(
                "The end value must be greater than the begin value but {}<={}".format(
                    end, begin))

        if self.num >= end - begin:
            raise ValueError(
                "The num value must be smaler than (end - begin) but {}>={}".format(
                    self.num, end - begin))

        subset_sizes = np.arange(begin, end, self.num, dtype=np.int64)

        list_of_answers = []
        points_one = np.ones(self.averaging, dtype=np.int64)

        if self.multiprocess:
            pool = Pool()
            mapping = pool.map
        else:
            mapping = map

        if self.progressbar:
            iterator = tqdm(subset_sizes)
        else:
            iterator = subset_sizes

        for m in iterator:
            list_of_answers.append(
                np.asarray(
                    list(mapping(self._score_subsample, m*points_one))))

        if self.multiprocess:
            pool.close()
            pool.join()

        list_of_answers = np.asarray(list_of_answers)

        list_of_E = np.mean(list_of_answers, axis = 1)
        list_of_S = np.std(list_of_answers, axis = 1)

        m_size = end
        for m, mean in zip(reversed(subset_sizes), reversed(list_of_E)):
            if mean > 1 - self.alpha:
                m_size = m

        return {'m*': m_size,
                'E': np.array(list_of_E),
                'S': np.array(list_of_S),
                'm': np.array(subset_sizes),
               }

class ALCEstimator(SampleSizeEstimator):
    r"""
    Description of ALC Method

    :param statmodel: the machine learning algorithm
    :type statmodel: RegressionModel or LogisticModel
    :param averaging: to do
    :type averaging: float
    :param alpha: to do
    :type alpha: float
    :param length: to do
    :type length: float
    :param begin: to do
    :type begin: int
    :param end: to do
    :type end: int
    :param num: to do
    :type num: int
    :param multiprocess: to do
    :type multiprocess: bool
    :param progressbar: to do
    :type progressbar: bool
    """

    def __init__(self, statmodel, **kwards):
        r"""Constructor method
        """
        self.statmodel = statmodel

        self.averaging = int(kwards.pop('averaging', 100))
        if self.averaging <= 0:
            raise ValueError(
                "The averaging should be positive but get {}".format(
                    self.averaging))

        self.length = kwards.pop('length', 0.5)
        if self.length <= 0:
            raise ValueError(
                "The length must be positive value but get {}".format(
                    self.length))

        self.alpha = kwards.pop('alpha', 0.05)
        if self.alpha < 0 or self.alpha > 1:
            raise ValueError(
                "The alpha must be between 0 and 1 but get {}".format(
                    self.alpha))
        
        self.begin = kwards.pop('begin', None)
        if self.begin is not None and self.begin < 0:
            raise ValueError(
                "The begin must be positive value but get {}".format(
                    self.begin))

        self.end = kwards.pop('end', None)
        if self.end is not None and self.end < 0:
            raise ValueError(
                "The end must be positive value but get {}".format(
                    self.end))

        if self.end is not None and self.begin is not None and self.end <= self.begin:
            raise ValueError(
                "The end value must be greater than the begin value but {}<={}".format(
                    self.end, self.begin))

        self.num = kwards.pop('num', 5)
        if self.num <=0:
            raise ValueError(
                "The num must be positive value but get {}".format(
                    self.num))
        if self.end is not None and self.begin is not None and self.num >= self.end - self.begin:
            raise ValueError(
                "The num value must be smaler than (end - begin) but {}>={}".format(
                    self.num, self.end - self.begin))

        self.multiprocess = kwards.pop('multiprocess', False)
        if not isinstance(self.multiprocess, bool):
            raise ValueError(
                "The multiprocess must be bool value but get {}".format(
                    self.multiprocess))

        self.progressbar = kwards.pop('progressbar', False)
        if not isinstance(self.progressbar, bool):
            raise ValueError(
                "The progressbar must be bool value but get {}".format(
                    self.progressbar))

        if kwards:
            raise ValueError("Invalid parameters: %s" % str(kwards))

        self.dataset = None

    def _aDistribution(self, dataset):
        r"""
        Return ...
        """
        X, y = dataset.sample()

        w_hat = self.statmodel(y, X).fit()

        cov = np.linalg.inv(
            0.01*np.eye(w_hat.shape[0]) - self.statmodel(y, X).hessian(w_hat))
        
        W = sps.multivariate_normal(mean=np.zeros(w_hat.shape[0]), cov = cov).rvs(size=1000)

        function = lambda r: np.abs( (np.sqrt((W**2).sum(axis=1)) > 3*r).mean() - self.alpha)
        return minimize_scalar(function, bounds=(0.01, 1), method='Bounded', options={'maxiter':10})['x']

    def _score_subsample(self, m):
        r"""
        Return ...
        """
        X_m, y_m = self.dataset.sample(m)
        dataset_m = Dataset(X_m, y_m)
        return self._aDistribution(dataset_m)

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

        self.dataset = Dataset(features, target)

        if self.end is None:
            end = len(self.dataset) - 1
        else:
            end = self.end

        if self.begin is None:
            begin = 2*self.dataset.n
        else:
            begin = self.begin

        if end <= begin:
            raise ValueError(
                "The end value must be greater than the begin value but {}<={}".format(
                    end, begin))

        if self.num >= end - begin:
            raise ValueError(
                "The num value must be smaler than (end - begin) but {}>={}".format(
                    self.num, end - begin))

        subset_sizes = np.arange(begin, end, self.num, dtype=np.int64)

        list_of_answers = []
        points_one = np.ones(self.averaging, dtype=np.int64)

        if self.multiprocess:
            pool = Pool()
            mapping = pool.map
        else:
            mapping = map

        if self.progressbar:
            iterator = tqdm(subset_sizes)
        else:
            iterator = subset_sizes

        for m in iterator:
            list_of_answers.append(
                np.asarray(
                    list(mapping(self._score_subsample, m*points_one))))

        if self.multiprocess:
            pool.close()
            pool.join()

        list_of_answers = np.asarray(list_of_answers)

        list_of_E = np.mean(list_of_answers, axis = 1)
        list_of_S = np.std(list_of_answers, axis = 1)

        m_size = end
        for m, mean in zip(reversed(subset_sizes), reversed(list_of_E)):
            if mean < self.length:
                m_size = m

        return {'m*': m_size,
                'E': np.array(list_of_E),
                'S': np.array(list_of_S),
                'm': np.array(subset_sizes),
               }

class MaxUtilityEstimator(SampleSizeEstimator):
    r"""
    Description of Utility Maximisation Method

    :param statmodel: the machine learning algorithm
    :type statmodel: RegressionModel or LogisticModel
    :param averaging: to do
    :type averaging: float
    :param c: to do
    :type c: float
    :param begin: to do
    :type begin: int
    :param end: to do
    :type end: int
    :param num: to do
    :type num: int
    :param multiprocess: to do
    :type multiprocess: bool
    :param progressbar: to do
    :type progressbar: bool
    """

    def __init__(self, statmodel, **kwards):
        r"""Constructor method
        """
        self.statmodel = statmodel

        self.averaging = int(kwards.pop('averaging', 100))
        if self.averaging <= 0:
            raise ValueError(
                "The averaging should be positive but get {}".format(
                    self.averaging))

        self.c = kwards.pop('c', 0.005)
        if self.c <= 0:
            raise ValueError(
                "The c must be positive value but get {}".format(
                    self.c))
        
        self.begin = kwards.pop('begin', None)
        if self.begin is not None and self.begin < 0:
            raise ValueError(
                "The begin must be positive value but get {}".format(
                    self.begin))

        self.end = kwards.pop('end', None)
        if self.end is not None and self.end < 0:
            raise ValueError(
                "The end must be positive value but get {}".format(
                    self.end))

        if self.end is not None and self.begin is not None and self.end <= self.begin:
            raise ValueError(
                "The end value must be greater than the begin value but {}<={}".format(
                    self.end, self.begin))

        self.num = kwards.pop('num', 5)
        if self.num <=0:
            raise ValueError(
                "The num must be positive value but get {}".format(
                    self.num))
        if self.end is not None and self.begin is not None and self.num >= self.end - self.begin:
            raise ValueError(
                "The num value must be smaler than (end - begin) but {}>={}".format(
                    self.num, self.end - self.begin))

        self.multiprocess = kwards.pop('multiprocess', False)
        if not isinstance(self.multiprocess, bool):
            raise ValueError(
                "The multiprocess must be bool value but get {}".format(
                    self.multiprocess))

        self.progressbar = kwards.pop('progressbar', False)
        if not isinstance(self.progressbar, bool):
            raise ValueError(
                "The progressbar must be bool value but get {}".format(
                    self.progressbar))

        if kwards:
            raise ValueError("Invalid parameters: %s" % str(kwards))

        self.dataset = None

    def _uFunction(self, dataset):
        r"""
        Return ...
        """
        X, y = dataset.sample()

        model = self.statmodel(y, X)
        w_hat = model.fit()

        cov = np.linalg.inv(
            0.01*np.eye(w_hat.shape[0]) - model.hessian(w_hat))
        
        prior = sps.multivariate_normal(mean = np.zeros(w_hat.shape[0]), cov = 0.01*np.eye(w_hat.shape[0]))

        W = sps.multivariate_normal(mean=w_hat, cov = cov).rvs(size=100)

        u = []
        for w in W:
            u.append(model.loglike(w) + prior.logpdf(w))

        return np.mean(u)/y.shape[0]  - self.c*y.shape[0]

    def _score_subsample(self, m):
        r"""
        Return ...
        """
        X_m, y_m = self.dataset.sample(m)
        dataset_m = Dataset(X_m, y_m)
        return self._uFunction(dataset_m)

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

        self.dataset = Dataset(features, target)

        if self.end is None:
            end = len(self.dataset) - 1
        else:
            end = self.end

        if self.begin is None:
            begin = 2*self.dataset.n
        else:
            begin = self.begin

        if end <= begin:
            raise ValueError(
                "The end value must be greater than the begin value but {}<={}".format(
                    end, begin))

        if self.num >= end - begin:
            raise ValueError(
                "The num value must be smaler than (end - begin) but {}>={}".format(
                    self.num, end - begin))

        subset_sizes = np.arange(begin, end, self.num, dtype=np.int64)

        list_of_answers = []
        points_one = np.ones(self.averaging, dtype=np.int64)

        if self.multiprocess:
            pool = Pool()
            mapping = pool.map
        else:
            mapping = map

        if self.progressbar:
            iterator = tqdm(subset_sizes)
        else:
            iterator = subset_sizes

        for m in iterator:
            list_of_answers.append(
                np.asarray(
                    list(mapping(self._score_subsample, m*points_one))))

        if self.multiprocess:
            pool.close()
            pool.join()

        list_of_answers = np.asarray(list_of_answers)

        list_of_E = np.mean(list_of_answers, axis = 1)
        list_of_S = np.std(list_of_answers, axis = 1)

        return {'m*': subset_sizes[np.argmax(np.array(list_of_E))],
                'E': np.array(list_of_E),
                'S': np.array(list_of_S),
                'm': np.array(subset_sizes),
               }


class KLEstimator(SampleSizeEstimator):
    r"""
    Description of KL based Method

    :param statmodel: the machine learning algorithm
    :type statmodel: RegressionModel or LogisticModel
    :param averaging: to do
    :type averaging: float
    :param epsilon: to do
    :type epsilon: float
    :param begin: to do
    :type begin: int
    :param end: to do
    :type end: int
    :param num: to do
    :type num: int
    :param multiprocess: to do
    :type multiprocess: bool
    :param progressbar: to do
    :type progressbar: bool
    """

    def __init__(self, statmodel, **kwards):
        r"""Constructor method
        """
        self.statmodel = statmodel

        self.averaging = int(kwards.pop('averaging', 5))
        if self.averaging <= 0:
            raise ValueError(
                "The averaging should be positive but get {}".format(
                    self.averaging))

        self.epsilon = kwards.pop('epsilon', 0.01)
        if self.epsilon <= 0:
            raise ValueError(
                "The epsilon must be positive value but get {}".format(
                    self.epsilon))
        
        self.begin = kwards.pop('begin', None)
        if self.begin is not None and self.begin < 0:
            raise ValueError(
                "The begin must be positive value but get {}".format(
                    self.begin))

        self.end = kwards.pop('end', None)
        if self.end is not None and self.end < 0:
            raise ValueError(
                "The end must be positive value but get {}".format(
                    self.end))

        if self.end is not None and self.begin is not None and self.end <= self.begin:
            raise ValueError(
                "The end value must be greater than the begin value but {}<={}".format(
                    self.end, self.begin))

        self.num = kwards.pop('num', 5)
        if self.num <=0:
            raise ValueError(
                "The num must be positive value but get {}".format(
                    self.num))
        if self.end is not None and self.begin is not None and self.num >= self.end - self.begin:
            raise ValueError(
                "The num value must be smaler than (end - begin) but {}>={}".format(
                    self.num, self.end - self.begin))

        self.multiprocess = kwards.pop('multiprocess', False)
        if not isinstance(self.multiprocess, bool):
            raise ValueError(
                "The multiprocess must be bool value but get {}".format(
                    self.multiprocess))

        self.progressbar = kwards.pop('progressbar', False)
        if not isinstance(self.progressbar, bool):
            raise ValueError(
                "The progressbar must be bool value but get {}".format(
                    self.progressbar))

        if kwards:
            raise ValueError("Invalid parameters: %s" % str(kwards))

        self.dataset = None

    @staticmethod
    def D_KL_normal(m_0, cov_0, m_1, cov_1, cov_0_inv, cov_1_inv):
        m_0 = np.array(m_0, ndmin=1)
        m_1 = np.array(m_1, ndmin=1)
        cov_0 = np.array(cov_0, ndmin=2)
        cov_1 = np.array(cov_1, ndmin=2)
        
        D_KL_1 = np.sum(np.diagonal(cov_1@cov_0_inv))
        D_KL_2 = float(np.reshape((m_1 - m_0), [1, -1])@cov_1@np.reshape((m_1 - m_0), [-1, 1]))
        D_KL_3 = -m_0.shape[0]
        D_KL_4 = float(np.log(np.linalg.det(cov_0)/np.linalg.det(cov_1)))
        
        return 0.5*(D_KL_1 + D_KL_2 + D_KL_3 + D_KL_4)

    def _klFunction(self, dataset):
        r"""
        Return ...
        """
        X, y = dataset.sample()

        model_0 = self.statmodel(y, X)
        m_0 = model_0.fit()
        cov_0_inv = 0.01*np.eye(m_0.shape[0]) - model_0.hessian(m_0)
        cov_0 = np.linalg.inv(cov_0_inv)

        # ind = np.random.randint(0, X.shape[0])
        indexes = np.random.permutation(X.shape[0])

        list_of_res = []

        for ind in indexes:
            X_new = np.delete(X, ind, axis = 0)
            y_new = np.delete(y, ind, axis = 0)

            model_1 = self.statmodel(y_new, X_new)
            m_1 = model_1.fit()
            cov_1_inv = 0.01*np.eye(m_1.shape[0]) - model_1.hessian(m_1)
            cov_1 = np.linalg.inv(cov_1_inv)
            list_of_res.append(
                self.D_KL_normal(m_0, cov_0, m_1, cov_1, cov_0_inv, cov_1_inv))

        return np.mean(list_of_res)


    def _score_subsample(self, m):
        r"""
        Return ...
        """
        X_m, y_m = self.dataset.sample(m)
        dataset_m = Dataset(X_m, y_m)
        return self._klFunction(dataset_m)

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

        self.dataset = Dataset(features, target)

        if self.end is None:
            end = len(self.dataset) - 1
        else:
            end = self.end

        if self.begin is None:
            begin = 2*self.dataset.n
        else:
            begin = self.begin

        if end <= begin:
            raise ValueError(
                "The end value must be greater than the begin value but {}<={}".format(
                    end, begin))

        if self.num >= end - begin:
            raise ValueError(
                "The num value must be smaler than (end - begin) but {}>={}".format(
                    self.num, end - begin))

        subset_sizes = np.arange(begin, end, self.num, dtype=np.int64)

        list_of_answers = []
        points_one = np.ones(self.averaging, dtype=np.int64)

        if self.multiprocess:
            pool = Pool()
            mapping = pool.map
        else:
            mapping = map

        if self.progressbar:
            iterator = tqdm(subset_sizes)
        else:
            iterator = subset_sizes

        for m in iterator:
            list_of_answers.append(
                np.asarray(
                    list(mapping(self._score_subsample, m*points_one))))

        if self.multiprocess:
            pool.close()
            pool.join()

        list_of_answers = np.asarray(list_of_answers)

        list_of_E = np.mean(list_of_answers, axis = 1)
        list_of_S = np.std(list_of_answers, axis = 1)

        m_size = end
        for m, mean in zip(reversed(subset_sizes), reversed(list_of_E)):
            if mean < self.epsilon:
                m_size = m

        return {'m*': m_size,
                'E': np.array(list_of_E),
                'S': np.array(list_of_S),
                'm': np.array(subset_sizes),
               }
