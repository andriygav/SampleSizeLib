#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`samplesizelib.utilily.api` contains classes:
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

import numpy as np

from samplesizelib.linear.statistical import LagrangeEstimator
from samplesizelib.linear.statistical import LikelihoodRatioEstimator
from samplesizelib.linear.statistical import WaldEstimator
from samplesizelib.linear.heuristic import CrossValidationEstimator
from samplesizelib.linear.heuristic import BootstrapEstimator
from samplesizelib.linear.heuristic import LogisticRegressionEstimator
from samplesizelib.linear.bayesian import APVCEstimator
from samplesizelib.linear.bayesian import ACCEstimator
from samplesizelib.linear.bayesian import ALCEstimator
from samplesizelib.linear.bayesian import MaxUtilityEstimator
from samplesizelib.linear.bayesian import KLEstimator
from samplesizelib.linear.models import LogisticModel, RegressionModel

NAME_TO_MODEL = {'LagrangeEstimator': LagrangeEstimator, 
                 'LikelihoodRatioEstimator': LikelihoodRatioEstimator, 
                 'WaldEstimator': WaldEstimator, 
                 'CrossValidationEstimator': CrossValidationEstimator, 
                 'BootstrapEstimator': BootstrapEstimator, 
                 'LogisticRegressionEstimator': LogisticRegressionEstimator, 
                 'APVCEstimator': APVCEstimator, 
                 'ACCEstimator': ACCEstimator, 
                 'ALCEstimator': ALCEstimator, 
                 'ALCEstimator': ALCEstimator, 
                 'MaxUtilityEstimator': MaxUtilityEstimator, 
                 'KLEstimator': KLEstimator}

NAME_TO_STATMODEL = {'LogisticModel': LogisticModel, 
                     'RegressionModel': RegressionModel}

def get_config():
    config = dict()
    config['LagrangeEstimator'] = {'epsilon': 0.3, 
                                   'alpha': 0.05, 
                                   'beta': 0.05}
    config['LikelihoodRatioEstimator'] = {'epsilon': 0.3, 
                                          'alpha': 0.05, 
                                          'beta': 0.05}
    config['WaldEstimator'] = {'epsilon': 0.3, 
                               'alpha': 0.05, 
                               'beta': 0.05}

    config['BootstrapEstimator'] = {'averaging': 100,
                                    'epsilon': 0.5, 
                                    'begin': None, 
                                    'end': None, 
                                    'num': 5}
    config['CrossValidationEstimator'] = {'averaging': 100, 
                                          'test_size': 0.5, 
                                          'epsilon': 0.05, 
                                          'begin': None, 
                                          'end': None, 
                                          'num': 5}

    config['APVCEstimator'] = {'averaging': 100,
                               'epsilon': 0.5, 
                               'begin': None, 
                               'end': None, 
                               'num': 5}
    config['ACCEstimator'] = {'averaging': 100,
                              'length': 0.25,
                              'alpha': 0.05,
                              'begin': None, 
                              'end': None, 
                              'num': 5}
    config['ALCEstimator'] = {'averaging': 100,
                              'length': 0.5,
                              'alpha': 0.05,
                              'begin': None, 
                              'end': None, 
                              'num': 5}
    config['MaxUtilityEstimator'] = {'averaging': 100,
                                     'c': 0.5, 
                                     'begin': None, 
                                     'end': None, 
                                     'num': 5}
    config['KLEstimator'] = {'averaging': 100,
                             'epsilon': 0.5, 
                             'begin': None, 
                             'end': None, 
                             'num': 5}
    return config
        