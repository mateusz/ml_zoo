import tensorflow as tf
import matplotlib
import numpy as np
from common import lib
from common import models

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def mkdata(words):
    mapped = dict(enumerate(words))
    hash = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(list(mapped.values())),
            tf.constant(list(mapped.keys()))
        ),
        default_value=-1
    )

    def vec(inputs):
        return tf.cast(hash[tf.constant(inputs)], tf.float32)

    def f(inputs):
        return (vec(inputs)/len(words) * 2.0 - 1.0)**2 * 5.0

    x = np.random.choice(words, size=100, replace=True)
    y = f(x) + tf.random.normal(shape=[100], stddev=0.5)

    return [lib.PlotData(words,f(words)).set_examples(x,y),vec]

def plot(
    pd: lib.PlotData,
    vec,
    model,
    name
):

    hvars = pd.hvars
    pad = hvars[-1]
    for _ in range(0, 10, 1):
        hvars.append(pad)

    losses = pd.losses
    pad = losses[-1]
    for _ in range(0, 10, 1):
        losses.append(pad)

    fig, ax = plt.subplots()

    def update(frame):
        z = zip(hvars[frame][1], model.trainable_variables)
        for a,v in z:
            v.assign(a)

        xground_embeds = model.do_embed(vec(pd.xground))
        x_embeds = model.do_embed(vec(pd.x))

        ax.clear()
        ax.scatter(xground_embeds, pd.yground, label='ground truth')
        for i, word in enumerate(pd.xground):
            ax.annotate(word, (xground_embeds[i], pd.yground[i]))
        ax.scatter(x_embeds, pd.y, marker='+', alpha=0.3, label='x')

        glued = np.vstack((xground_embeds[:,0], model(vec(pd.xground))[:,0,0])).transpose()
        glued = glued[glued[:,0].argsort()]
        ax.plot(glued[:,0], glued[:,1], label='model', color='darkgreen')
        ax.text(0.82, 0.01, 'E#%d L=%.1f' % (hvars[frame][0], losses[frame]), transform=ax.transAxes)

        ax.legend()

    ani = FuncAnimation(fig, update, frames=np.arange(0, len(hvars), 1))
    ani.save('dvcanimations/%s.gif' % name, writer=PillowWriter(fps=5))
    plt.close()


def main():
    words = ['big', 'red', 'robot', 'jumped', 'over', 'a', 'lazy', 'ai']
    pd,vec = mkdata(words)

    m = models.WithEmbed(len(words), 1)
    dataset = tf.data.Dataset.from_tensor_slices((vec(pd.x), pd.y))
    dataset = dataset.shuffle(buffer_size=pd.x.shape[0]).batch(16)
    lib.sgd(pd, m, dataset, lib.mse_loss, learning_rate=0.01, epochs=20)
    plot(pd,vec,m,'nlp_embed')

if __name__=='__main__':
    main()