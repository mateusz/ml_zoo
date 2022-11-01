import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

def mkdata():
    x = tf.linspace(-2, 2, 201)
    x = tf.cast(x, tf.float32)

    def f(x):
        return x**2 + 2*x - 5
        
    y = f(x) + tf.random.normal(shape=[201], stddev=0.5)

    return [x,y,f(x)]

class HandmadeTinyRelu(tf.Module):

  def __init__(self):
    self.w = tf.Variable(tf.random.normal(shape=[1], stddev=1.0))
    self.b = tf.Variable([0.0])
    self.w2 = tf.Variable(tf.random.normal(shape=[1], stddev=1.0))
    self.b2 = tf.Variable([0.0])

  @tf.function
  def __call__(self, x):
    x = tf.stack([x], axis=1)

    x = x*self.w + self.b
    x = tf.nn.relu(x)
    x = x*self.w2 + self.b2

    return x

def mse_loss(y_pred, y):
  return tf.reduce_mean(tf.square(y_pred - y))

def sgd(
    model,
    dataset,
    loss_fn,
    epochs = 20,
    learning_rate = 0.01,
    skip_rate = 1
):
    hvars = []
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
            loss = mean_epoch_loss.result()
            print('%d MSE=%.2f' % (epoch, mean_epoch_loss.result()))
            hvars.append([epoch, [tf.constant(v) for v in model.trainable_variables]])

    return hvars

def plot(x,y,yorig,hvars,model):
    fig, ax = plt.subplots()
    ax.plot(x, y, '.', label='x')
    ax.plot(x, yorig, label='ground truth')
    mplot, = ax.plot([], [], label='model')
    frame_no = ax.text(0.82, 0.01, '', transform=ax.transAxes)
    ax.legend()

    def init():
        mplot.set_data([], [])
        frame_no.set_text('')
        return tuple([mplot]) + tuple([frame_no])

    def update(frame):
        z = zip(hvars[frame][1], model.trainable_variables)
        for a,v in z:
            v.assign(a)
        mplot.set_data(x, model(x))
        frame_no.set_text('Epoch #%d' % hvars[frame][0])
        return tuple([mplot]) + tuple([frame_no])

    ani = FuncAnimation(fig, update, init_func=init, frames=np.arange(0, len(hvars), 1), blit=True, repeat=False)
    ani.save('dvcanimations/ff_tiny_relu.gif', writer=PillowWriter(fps=5))
    plt.close()

def main():
    x,y,yorig = mkdata()
    m = HandmadeTinyRelu()
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=x.shape[0]).batch(16)
    hvars = sgd(m, dataset, mse_loss)
    plot(x,y,yorig,hvars,m)

if __name__=='__main__':
    main()