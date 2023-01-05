import tensorflow as tf
import matplotlib
import numpy as np
from common import lib
from common import models


def mkdata():
    x = tf.linspace(-2, 2, 201)
    x = tf.cast(x, tf.float32)

    def f(x):
        return (3*x**3 + x**2 + 2*x - 5 + 7*tf.sin(5*x))/4.0

    y = f(x) + tf.random.normal(shape=[201], stddev=0.5)

    return lib.PlotData(x,f(x)).set_examples(x,y)


def main():
    pd = mkdata()
    m = models.DenseRelu('relu1024', 1024)
    dataset = tf.data.Dataset.from_tensor_slices((pd.x, pd.y))
    dataset = dataset.shuffle(buffer_size=pd.x.shape[0]).batch(16)
    lib.sgd(pd, m, dataset, lib.mse_loss, learning_rate=0.0005, epochs=40)
    lib.plot(pd, name='ff_periodic')

if __name__=='__main__':
    main()