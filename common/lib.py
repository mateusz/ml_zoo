import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def sgd(
    model,
    dataset,
    loss_fn,
    epochs = 20,
    learning_rate = 0.01,
    skip_rate = 1
):
    hvars = []
    losses = []
    for epoch in range(epochs):
        mean_epoch_loss = tf.metrics.Mean()
        for x, y in dataset:
            with tf.GradientTape() as tape:
                ypred = model(x)
                yloss = loss_fn(ypred, tf.stack([y], axis=1))

            grads = tape.gradient(yloss, model.trainable_variables)
            mean_epoch_loss.update_state(yloss)

            for g,v in zip(grads, model.trainable_variables):
                v.assign_sub(learning_rate*g)

        if epoch % skip_rate == 0:
            l = mean_epoch_loss.result().numpy()
            print('%d MSE=%.2f' % (epoch, l))
            losses.append(l)
            hvars.append([epoch, [tf.constant(v) for v in model.trainable_variables]])

    return hvars, losses

def plot(x,y,yorig,hvars,losses,model,name):
    fig, ax = plt.subplots()
    ax.plot(x, y, '.', label='x')
    ax.plot(x, yorig, label='ground truth')
    mplot, = ax.plot([], [], label='model')
    info = ax.text(0.82, 0.01, '', transform=ax.transAxes)
    ax.legend()

    pad = hvars[-1]
    for _ in range(0, 10, 1):
        hvars.append(pad)
    pad = losses[-1]
    for _ in range(0, 10, 1):
        losses.append(pad)

    def init():
        mplot.set_data([], [])
        info.set_text('')
        return tuple([mplot]) + tuple([info])

    def update(frame):
        z = zip(hvars[frame][1], model.trainable_variables)
        for a,v in z:
            v.assign(a)
        mplot.set_data(x, model(x))
        info.set_text('E#%d L=%.1f' % (hvars[frame][0], losses[frame]))
        return tuple([mplot]) + tuple([info])

    ani = FuncAnimation(fig, update, init_func=init, frames=np.arange(0, len(hvars), 1), blit=True)
    ani.save('dvcanimations/%s.gif' % name, writer=PillowWriter(fps=5))
    plt.close()

def mse_loss(y_pred, y):
  return tf.reduce_mean(tf.square(y_pred - y))