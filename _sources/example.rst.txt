*******
Example
*******

Requirements
============

It is recommended make virtualenv and install all next packages
in this virtualenv.

::

    samplesizelib==0.0.2

Include packages.

.. code:: python
    
    import numpy as np

    from samplesizelib.linear.statistical import LagrangeEstimator
    from samplesizelib.linear.statistical import LikelihoodRatioEstimator
    from samplesizelib.linear.statistical import WaldEstimator
    from samplesizelib.linear.models import RegressionModel
    from samplesizelib.linear.models import LogisticModel

    from samplesizelib.linear.heuristic import CrossValidationEstimator
    from samplesizelib.linear.heuristic import BootstrapEstimator
    from samplesizelib.linear.heuristic import LogisticRegressionEstimator
    from samplesizelib.linear.bayesian import APVCEstimator
    from samplesizelib.linear.bayesian import ACCEstimator
    from samplesizelib.linear.bayesian import ALCEstimator
    from samplesizelib.linear.bayesian import MaxUtilityEstimator
    from samplesizelib.linear.bayesian import KLEstimator



Preparing the dataset
=====================

Generate dataset for regression and classification tasks.

.. code:: python

    n = 10
    m = 300

    np.random.seed(0)
    X_cl = np.random.randn(m, n)
    y_cl = np.random.randint(2, size=m)

    np.random.seed(0)
    X_rg = np.random.randn(m, n)
    y_rg = np.random.randn(m)

Bayesian Metods
===============

Regression task
---------------

Example of Bootstrap based method:

.. code:: python

    model = BootstrapEstimator(RegressionModel)
    ret = model(X_rg, y_rg)

    print(ret['m*'])

Example of Cross Validation based method:

.. code:: python

    model = CrossValidationEstimator(RegressionModel)
    ret = model(X_rg, y_rg)

    print(ret['m*'])


Classification task
-------------------

Example of Logistic Regression method:

.. code:: python

    model = LogisticRegressionEstimator(LogisticModel)
    ret = model(X_cl, y_cl)

    print(ret['m*'])

Example of Bootstrap based method:

.. code:: python

    model = BootstrapEstimator(LogisticModel)
    ret = model(X_cl, y_cl)

    print(ret['m*'])

Example of Cross Validation based method:

.. code:: python

    model = CrossValidationEstimator(LogisticModel)
    ret = model(X_cl, y_cl)

    print(ret['m*'])


Bayesian Metods
===============

Regression task
---------------

Example of KL-divergence method:

.. code:: python

    model = KLEstimator(RegressionModel)
    ret = model(X_rg, y_rg)

    print(ret['m*'])

Example of Max Utility method:

.. code:: python

    model = MaxUtilityEstimator(RegressionModel)
    ret = model(X_rg, y_rg)

    print(ret['m*'])

Example of ALC method:

.. code:: python

    model = ALCEstimator(RegressionModel)
    ret = model(X_rg, y_rg)

    print(ret['m*'])

Example of ACC method:

.. code:: python

    model = ACCEstimator(RegressionModel)
    ret = model(X_rg, y_rg)

    print(ret['m*'])

Example of APVC method:

.. code:: python

    model = APVCEstimator(RegressionModel)
    ret = model(X_rg, y_rg)

    print(ret['m*'])

Classification task
-------------------

Example of KL-divergence method:

.. code:: python

    model = KLEstimator(LogisticModel)
    ret = model(X_cl, y_cl)

    print(ret['m*'])

Example of Max Utility method:

.. code:: python

    model = MaxUtilityEstimator(LogisticModel)
    ret = model(X_cl, y_cl)

    print(ret['m*'])

Example of ALC method:

.. code:: python

    model = ALCEstimator(LogisticModel)
    ret = model(X_cl, y_cl)

    print(ret['m*'])

Example of ACC method:

.. code:: python

    model = ACCEstimator(LogisticModel)
    ret = model(X_cl, y_cl)

    print(ret['m*'])

Example of APVC method:

.. code:: python

    model = APVCEstimator(LogisticModel)
    ret = model(X_cl, y_cl)

    print(ret['m*'])

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
