import tensorflow as tf

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
    
class DenseRelu(tf.Module):

  def __init__(self, dense=16):
    self.l1 = tf.keras.layers.Dense(units=dense, activation='relu', bias_initializer=tf.random.normal, kernel_initializer=tf.random.normal)
    self.l1.build((1))
    self.l2 = tf.keras.layers.Dense(units=1, activation='linear', kernel_initializer=tf.random.normal, name='decision')
    self.l2.build((dense))

  @tf.function
  def __call__(self, x):
    x = tf.stack([x], axis=1)
    x = self.l1(x)
    x = self.l2(x)
    return x