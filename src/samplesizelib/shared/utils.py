#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The :mod:`samplesizelib.shared.utils` contains classes:
"""
from __future__ import print_function

__docformat__ = 'restructuredtext'

import numpy as np

class Dataset(object):
    r"""
    """

    def __init__(self, X, y):
        r"""Constructor method
        """
        self._X = X
        self._y = y

        self.n = self._X.shape[1]

    def __len__(self):
        return len(self._X)


    def sample(self, m=None, duplications=True):
        r"""
        :param m: subset size. must be greater than number of feature
        :type m: int
        :param duplications: to do
        :type duplications: bool
        """
        if m is None:
            m = self.__len__()
        if m <= self.n:
            raise ValueError(
                "The m={} value must be greater than number of feature={}".format(
                    m, self.n))
        if duplications:
            indexes = np.random.randint(
                low = 0, high=self._X.shape[0], size = m)
        else:
            indexes = np.random.permutation(self._X.shape[0])[:m]
        
        X_m = self._X[indexes, :]
        y_m = self._y[indexes]
        while ((y_m == 0).sum() > m - 2 or (y_m == 1).sum() > m - 2):
            if duplications:
                indexes = np.random.randint(
                    low = 0, high=self._X.shape[0], size = m)
            else:
                indexes = np.random.permutation(X.shape[0])[:m]
        
            X_m = self._X[indexes, :]
            y_m = self._y[indexes]
        return X_m, y_m

    def train_test_split(self, test_size = 0.5, safe=True):
        r"""

        """
        X = self._X
        y = self._y

        M = int(X.shape[0]*test_size)
        indexes_test = np.random.permutation(X.shape[0])[:M]
        indexes_train = np.random.permutation(X.shape[0])[M:]
        X_train = X[indexes_train, :]
        X_test = X[indexes_test, :]
        y_train = y[indexes_train]
        y_test = y[indexes_test]
        if safe:
            while ((y_train == 0).all() or (y_train == 1).all() or (y_test == 0).all() or (y_test == 1).all()):
                indexes_test = np.random.permutation(X.shape[0])[:M]
                indexes_train = np.random.permutation(X.shape[0])[M:]
                X_train = X[indexes_train, :]
                X_test = X[indexes_test, :]
                y_train = y[indexes_train]
                y_test = y[indexes_test]
        return X_train, X_test, y_train, y_test

