import numpy as np

from samplesizelib.linear.statistical import LagrangeEstimator, LikelihoodRatioEstimator, WaldEstimator
from samplesizelib.linear.models import RegressionModel, LogisticModel

def test_classification():
    n = 10
    m = 200

    np.random.seed(0)
    X = np.random.randn(m, n)
    y = np.random.randint(2, size=m)

    model = LagrangeEstimator(LogisticModel)
    ret = model(X, y)

    assert ret['m*'] == 109

def test_regression():
    n = 10
    m = 200

    np.random.seed(0)
    X = np.random.randn(m, n)
    y = np.random.randn(m)

    model = LagrangeEstimator(RegressionModel)
    ret = model(X, y)

    assert ret['m*'] == 23
