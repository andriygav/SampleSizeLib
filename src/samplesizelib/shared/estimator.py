#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`samplesizelib.shared.estimator` contains classes:
- :class:`samplesizelib.shared.estimator.SampleSizeEstimator`
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

class SampleSizeEstimator(object):
    r"""Base class for all sample size estimation models."""

    def __init__(self):
        r"""Constructor method
        """
        pass

    def __call__(self, features, target):
        r"""
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
        :rtype: float
        """
        raise NotImplementedError