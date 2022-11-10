import tensorflow as tf
import matplotlib
import numpy as np
from common import lib
from common import models


def mkdata():
    x = tf.linspace(-2, 2, 201)
    x = tf.cast(x, tf.float32)

    def f(x):
        return (4*x**3 + x**2 + 2*x - 5)/5.0
        
    y = f(x) + tf.random.normal(shape=[201], stddev=0.5)

    return lib.PlotData(x,f(x)).set_examples(x,y)


def main():
    pd = mkdata()
    m = models.DenseRelu(32)
    dataset = tf.data.Dataset.from_tensor_slices((pd.x, pd.y))
    dataset = dataset.shuffle(buffer_size=pd.x.shape[0]).batch(16)
    lib.sgd(pd, m, dataset, lib.mse_loss, learning_rate=0.005, epochs=40)
    lib.plot(pd, m, name='ff_cubic')

if __name__=='__main__':
    main()