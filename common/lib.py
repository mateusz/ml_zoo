import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

class PlotData:
    def __init__(
        self,
        xground,
        yground,
    ):
        self.xground = xground
        self.yground = yground

        self.x = []
        self.y = []

        self.models = {}
        self.hvars = {}
        self.losses = {}

    def set_examples(self, x, y):
        self.x = x
        self.y = y
        return self

    def add_model(self, id, model):
        self.models[id] = model

    def add_hvar(self, id, hv):
        if id not in self.hvars.keys():
            self.hvars[id] = []
        self.hvars[id].append(hv)
        return self

    def add_loss(self, id, l):
        if id not in self.losses.keys():
            self.losses[id] = []
        self.losses[id].append(l)
        return self

def sgd(
    pd: PlotData,
    model,
    dataset,
    loss_fn,
    epochs = 20,
    learning_rate = 0.01,
    skip_rate = 1
):
    pd.add_model(model.name, model)
    pd.add_loss(model.name, np.Inf)
    pd.add_hvar(model.name, [0, [tf.constant(v) for v in model.trainable_variables]])
    for epoch in range(1, epochs+1):
        mean_epoch_loss = tf.metrics.Mean()
        for x, y in dataset:
            with tf.GradientTape() as tape:
                ypred = model(x)
                yloss = loss_fn(ypred, tf.stack([y], axis=1))

            grads = tape.gradient(yloss, model.trainable_variables)
            mean_epoch_loss.update_state(yloss)

            for g,v in zip(grads, model.trainable_variables):
                if type(g) is tf.IndexedSlices:
                    v.assign_sub(learning_rate*tf.convert_to_tensor(g))
                else:
                    v.assign_sub(learning_rate*g)

        if epoch % skip_rate == 0:
            l = mean_epoch_loss.result().numpy()
            print('%d loss=%.2f' % (epoch, l))
            pd.add_loss(model.name, l)
            pd.add_hvar(model.name, [epoch, [tf.constant(v) for v in model.trainable_variables]])

def copy_model(dst,src):
    z = zip(src.trainable_variables, dst.trainable_variables)
    for s,d in z:
        d.assign(s)

def plot(
    pd: PlotData,
    name
):
    fig, ax = plt.subplots()
    ax.plot(pd.x, pd.y, '.', label='x')
    ax.plot(pd.xground, pd.yground, label='ground truth', color='black', linewidth=3.0, linestyle='dashed')
    mplots = {}
    for k in pd.models.keys():
        mplots[k], = ax.plot([], [], label=pd.models[k].name)

    info = ax.text(0.9, 0.01, '', transform=ax.transAxes)
    ax.legend()

    frames = 0
    # TODO pad to longest
    hvars = pd.hvars
    for k in hvars.keys():
        pad = hvars[k][-1]
        for _ in range(0, 10, 1):
            hvars[k].append(pad)
        frames = len(hvars[k])

    losses = pd.losses
    for k in losses.keys():
        pad = losses[k][-1]
        for _ in range(0, 10, 1):
            losses[k].append(pad)

    def init():
        for k in mplots.keys():
            mplots[k].set_data([], [])
        info.set_text('')
        return tuple(list(mplots.values())) + tuple([info])

    def update(frame):
        for k in mplots.keys():
            z = zip(hvars[k][frame][1], pd.models[k].trainable_variables)
            for a,v in z:
                v.assign(a)
            mplots[k].set_data(pd.xground, pd.models[k](pd.xground))
        info.set_text('E#%d' % (hvars[k][frame][0]))
        return tuple(list(mplots.values())) + tuple([info])

    ani = FuncAnimation(fig, update, init_func=init, frames=np.arange(0, frames, 1), blit=True)
    ani.save('dvcanimations/%s.gif' % name, writer=PillowWriter(fps=5))
    plt.close()

def mse_loss(y_pred, y):
    return tf.reduce_mean(tf.square(y_pred - y))

def mae_loss(y_pred, y):
    return tf.reduce_mean(tf.abs(y_pred - y))

def msle_loss(y_pred, y):
    y_pred_log = tf.math.log(tf.maximum(y_pred, tf.keras.backend.epsilon()) + 1.0)
    y_log = tf.math.log(tf.maximum(y, tf.keras.backend.epsilon()) + 1.0)
    return tf.reduce_mean(tf.square(y_pred_log - y_log)) 

def log_loss(y_pred, y):
    # Calculated from negative log likelihood on top of exponential distribution.
    # Does not work as a loss for exponentially-distributed ground truths :-)
    # Unless I'm wrong, which is possible, this shows motivating MSE using
    # negative log likelihood of Gaussian is valid, but contrived.
    return tf.reduce_mean(y_pred/y - tf.math.log(1/y))

# https://github.com/lukovkin/linex-keras
# Use a < 0 to penalize errors with negative values more, and a > 0 otherwise.
class linex_loss:
    def __init__(self, a=-1, b=1):
        self.a = a
        self.b = b

    def __call__(self, y_pred, y):
        # b * (exp(a * x) - a * x - 1)
        delta = tf.math.abs(y_pred - y)
        return tf.reduce_mean(self.b * (tf.math.exp(self.a*delta)-self.a*delta-1))
