import tensorflow as tf
import matplotlib
import numpy as np
from common import lib
from common import models
import os, random, copy

def mkdata():
    
    x = tf.linspace(-2, 2, 201)
    x = tf.cast(x, tf.float32)

    def f(x):
        return 2*x**2 + 2*x + 2

    y = f(x) + tf.random.normal(shape=[201], stddev=0.5)

    return lib.PlotData(x,f(x)).set_examples(x,y)


def main():
    pd = mkdata()
    dataset = tf.data.Dataset.from_tensor_slices((pd.x, pd.y))
    dataset = dataset.shuffle(buffer_size=pd.x.shape[0]).batch(16)

    m = models.DenseRelu('base', 16)
    m_pos = models.DenseRelu('a_positive', 16)
    m_neg = models.DenseRelu('a_negative', 16)
    m_neg_hi = models.DenseRelu('a_negative_hi', 16)
    lib.copy_model(m_pos, m)
    lib.copy_model(m_neg, m)
    lib.copy_model(m_neg_hi, m)

    lib.sgd(pd, m_pos, dataset, lib.linex_loss(a=0.2), learning_rate=0.01, epochs=40)
    lib.sgd(pd, m_neg, dataset, lib.linex_loss(a=-0.2), learning_rate=0.01, epochs=40)
    lib.sgd(pd, m_neg_hi, dataset, lib.linex_loss(a=-0.4), learning_rate=0.01, epochs=40)

    lib.plot(pd, 'ff_quadratic_linex')

if __name__=='__main__':
    main()