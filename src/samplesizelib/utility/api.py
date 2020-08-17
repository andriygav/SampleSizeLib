#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`samplesizelib.utilily.api` contains classes:
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

import numpy as np

from ..linear.statistical import LagrangeEstimator
from ..linear.statistical import LikelihoodRatioEstimator
from ..linear.statistical import WaldEstimator
from ..linear.heuristic import CrossValidationEstimator
from ..linear.heuristic import BootstrapEstimator
from ..linear.heuristic import LogisticRegressionEstimator
from ..linear.bayesian import APVCEstimator
from ..linear.bayesian import ACCEstimator
from ..linear.bayesian import ALCEstimator
from ..linear.bayesian import MaxUtilityEstimator
from ..linear.bayesian import KLEstimator



class LinearSampleSizeEstimator(object):
    r"""
    Class for analyse datased by all methods.

    :param statmodel: the machine learning algorithm
    :type statmodel: RegressionModel or LogisticModel
    """

    def __init__(self, statmodel, **kwards):
        r"""Constructor method
        """
        self.statmodel = statmodel

        self.models = dict()

        self.models['statistical'] = dict()
        self.models['heuristic'] = dict()
        self.models['bayesian'] = dict()

        self.models['statistical']['LagrangeEstimator'] = LagrangeEstimator(statmodel)
        self.models['statistical']['LikelihoodRatioEstimator'] = LikelihoodRatioEstimator(statmodel)
        self.models['statistical']['WaldEstimator'] = WaldEstimator(statmodel)

        self.models['heuristic']['BootstrapEstimator'] = BootstrapEstimator(statmodel)
        self.models['heuristic']['CrossValidationEstimator'] = CrossValidationEstimator(statmodel)

        self.models['bayesian']['APVCEstimator'] = APVCEstimator(statmodel)
        self.models['bayesian']['ACCEstimator'] = ACCEstimator(statmodel)
        self.models['bayesian']['ALCEstimator'] = ALCEstimator(statmodel)
        self.models['bayesian']['MaxUtilityEstimator'] = MaxUtilityEstimator(statmodel)
        self.models['bayesian']['KLEstimator'] = KLEstimator(statmodel)

    def __call__(self, features, target):
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
        return self.forward(features, target)

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

        result = dict()

        for key in self.models:
            result[key] = dict()
            for name in self.models[key]:
                result[key][name] = self.models[key][name](features, target)

        return result
        