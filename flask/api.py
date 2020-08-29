#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`samplesizelib.utilily.api` contains classes:
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

import numpy as np
from sklearn.preprocessing import scale

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
    config['LagrangeEstimator'] = {'epsilon': 0.2, 
                                   'alpha': 0.05, 
                                   'beta': 0.2}
    config['LikelihoodRatioEstimator'] = {'epsilon': 0.2, 
                                          'alpha': 0.05, 
                                          'beta': 0.2}
    config['WaldEstimator'] = {'epsilon': 0.2, 
                               'alpha': 0.05, 
                               'beta': 0.2}

    config['BootstrapEstimator'] = {'averaging': 10,
                                    'epsilon': 0.5}
    config['CrossValidationEstimator'] = {'averaging': 10,
                                          'test_size': 0.5, 
                                          'epsilon': 0.05}

    config['APVCEstimator'] = {'averaging': 10,
                               'epsilon': 0.5}
    config['ACCEstimator'] = {'averaging': 10,
                              'length': 0.25,
                              'alpha': 0.05}
    config['ALCEstimator'] = {'averaging': 10,
                              'length': 0.5,
                              'alpha': 0.05}
    config['MaxUtilityEstimator'] = {'averaging': 10,
                                     'c': 0.005}
    config['KLEstimator'] = {'averaging': 5,
                             'epsilon': 0.5}
    return config

class worker(object):
    def __init__(self, statmodel, config, X, y):
        self.config = config
        self.statmodel = statmodel
        self.X = X
        self.y = y

        self.progress = dict()

        self.models = dict()
        self.status = None
        self.result = dict()

        self._percentage_of_completion_status = 0.

        for key in config:
            try:
                self.models[key] = NAME_TO_MODEL[key](statmodel, **config[key])
                self.progress[key] = dict()
                self.progress[key]['name'] = key
                self.progress[key]['status'] = 'none'
            except ValueError as e:
                self.status = 'Model "{}" initialise error: {}'.format(key, str(e))
                return

    def percentage(self):
        r"""
        Returns the percentage of completion.
        
        :return: percentage of completion.
        :rtype: float
        """
        return self._percentage_of_completion_status

    def _set_percentage(self, new_percentage):
        r"""
        change percentage of completion status
        """
        new_percentage = float(new_percentage)
        if 0 <= new_percentage <= 100:
            self._percentage_of_completion_status = new_percentage


    def forward(self):
        for i, key in enumerate(self.models):
            self.progress[key] = dict()
            self.progress[key]['status'] = 'start'
            try:
                self.result[key] = self.models[key](self.X, self.y)
            except Exception as e:
                self.status = 'Model "{}" running error: {}'.format(key, str(e))
                self.progress[key]['status'] = 'error'
                return self.result
            self.progress[key]['status'] = 'end'
            
            self.progress[key]['result'] = self.result[key]
            self._set_percentage(100.*(i+1)/len(self.models))

        return self.result

class scheduler(object):
    def __init__(self):
        self.id_to_model = dict()

    def add_job(self, model):
        _id = hash(model)& 0xffffffff
        self.id_to_model[_id] = model
        return _id

    def get_job(self, _id):
        return self.id_to_model.get(_id, None)

        