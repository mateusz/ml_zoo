#%%

import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib
import numpy as np
from common import lib
from common import models
import scipy.stats as stats
import math

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
    m = models.DenseRelu(16)
    dataset = tf.data.Dataset.from_tensor_slices((pd.x, pd.y))
    dataset = dataset.shuffle(buffer_size=pd.x.shape[0]).batch(16)
    # https://www.reddit.com/r/MachineLearning/comments/9r79mr/d_what_kind_of_loss_function_is_best_fit_to_learn/
    # Tehse seems to be more stable and matches better
    lib.sgd(pd, m, dataset, lib.linex_loss(a=-0.2), learning_rate=0.1, epochs=20)
    # This is more unstable, and tends to overestimate
    #lib.sgd(pd, m, dataset, lib.mse_loss, learning_rate=0.001, epochs=20)
    lib.plot(pd, m, name='ff_quadratic_exp')

if __name__=='__main__':
    main()