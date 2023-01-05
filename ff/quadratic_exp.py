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

    y = []
    for o in x:
        s = stats.norm.rvs(loc=math.log(f(o)), scale=1, size=1)
        s = math.exp(s)
        y.append(tf.cast(s, tf.float32))

    return lib.PlotData(x,f(x)).set_examples(x,y)


def main():
    pd = mkdata()
    dataset = tf.data.Dataset.from_tensor_slices((pd.x, pd.y))
    dataset = dataset.shuffle(buffer_size=pd.x.shape[0]).batch(16)

    m = models.DenseRelu('base', 16)
    m_mse = models.DenseRelu('mse', 16)
    m_mae = models.DenseRelu('mae', 16)
    m_msle = models.DenseRelu('msle', 16)
    m_linex = models.DenseRelu('linex', 16)
    lib.copy_model(m_mse, m)
    lib.copy_model(m_mae, m)
    lib.copy_model(m_msle, m)
    lib.copy_model(m_linex, m)

    lib.sgd(pd, m_mse, dataset, lib.mse_loss, learning_rate=0.0005, epochs=40)
    lib.sgd(pd, m_mae, dataset, lib.mae_loss, learning_rate=0.005, epochs=40)
    lib.sgd(pd, m_msle, dataset, lib.msle_loss, learning_rate=0.05, epochs=40)
    lib.sgd(pd, m_linex, dataset, lib.linex_loss(a=-0.2), learning_rate=0.05, epochs=40)

    lib.plot(pd, 'ff_quadratic_exp')


if __name__=='__main__':
    main()