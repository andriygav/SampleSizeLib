*******
Example
*******

Requirements
============

It is recommended make virtualenv and install all next packages
in this virtualenv.

::

    samplesizelib==0.0.1

Include packages.

.. code:: python
    
    import numpy as np

    from samplesizelib.linear.statistical import LagrangeEstimator
    from samplesizelib.linear.statistical import LikelihoodRatioEstimator
    from samplesizelib.linear.statistical import WaldEstimator
    from samplesizelib.linear.models import RegressionModel
    from samplesizelib.linear.models LogisticModel

Preparing the dataset
=====================

Generate dataset for regression and classification tasks.

.. code:: python

    n = 10
    m = 200

    np.random.seed(0)
    X_cl = np.random.randn(m, n)
    y_cl = np.random.randint(2, size=m)

    np.random.seed(0)
    X_rg = np.random.randn(m, n)
    y_rg = np.random.randn(m)


Statictical Metods
==================

Regression task
---------------

Example of Lagrange based method:

.. code:: python

    model = LagrangeEstimator(RegressionModel)
    ret = model(X_rg, y_rg)

    print(ret['m*'])


Example of Likelihood Ratio based method:

.. code:: python

    model = LikelihoodRatioEstimator(RegressionModel)
    ret = model(X_rg, y_rg)

    print(ret['m*'])

Example of Wald based method:

.. code:: python

    model = WaldEstimator(RegressionModel)
    ret = model(X_rg, y_rg)

    print(ret['m*'])


Classification task
-------------------

Example of Lagrange based method:

.. code:: python

    model = LagrangeEstimator(LogisticModel)
    ret = model(X_cl, y_cl)

    print(ret['m*'])


Example of Likelihood Ratio based method:

.. code:: python

    model = LikelihoodRatioEstimator(LogisticModel)
    ret = model(X_cl, y_cl)

    print(ret['m*'])

Example of Wald based method:

.. code:: python

    model = WaldEstimator(LogisticModel)
    ret = model(X_cl, y_cl)

    print(ret['m*'])
