#%%

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib
import numpy as np
from common import lib
from common import models
import scipy.stats as stats
import math
import copy

def mkdata():

    x = tf.linspace(-2, 2, 201)
    x = tf.cast(x, tf.float32)

    def f(x):
        return 2*x**2 + 2*x + 2

    exp_mean = 5
    y = f(x) + tfp.distributions.Exponential(rate=1/exp_mean).sample(201)-exp_mean

    return lib.PlotData(x,f(x)).set_examples(x,y)


def main():
    pd = mkdata()
    dataset = tf.data.Dataset.from_tensor_slices((pd.x, pd.y))
    dataset = dataset.shuffle(buffer_size=pd.x.shape[0]).batch(16)

    m = models.DenseRelu('base', 16)
    m_exp = models.DenseRelu('mle_exp', 16)
    m_mse = models.DenseRelu('mse', 16)
    lib.copy_model(m_exp, m)
    lib.copy_model(m_mse, m)

    lib.sgd(pd, m_exp, dataset, lib.mle_exp_loss, learning_rate=0.1, epochs=40)
    lib.sgd(pd, m_mse, dataset, lib.mse_loss, learning_rate=0.0005, epochs=40)

    lib.plot(pd, 'ff_quadratic_exp')


if __name__=='__main__':
    main()