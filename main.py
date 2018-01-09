#!/usr/bin/env python3

"""
Simple example using LARS. In general, to use LARS you just need to change
the line of code creating your Optimizer.

Usage example:
  ./main.py
"""


import lars
import numpy as np
import random
import tensorflow as tf

RANDOM_SEED = 10725
TRAINING_STEPS = 10000
OUT_FREQ = 1000


def main():
  random.seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)
  tf.set_random_seed(RANDOM_SEED)

  answer = 123.0
  x_list = np.arange(-5, 5)
  y_list = answer * x_list

  X = tf.constant(x_list, dtype=tf.float32)
  Y = tf.constant(y_list, dtype=tf.float32)

  W = tf.Variable(tf.random_normal([1]), name='W')
  loss = tf.losses.mean_squared_error(Y, tf.multiply(X, W))
  train_step, _ = lars.createLarsMinimizer(
      loss=loss,
      initial_learning_rate=1.0,
      learning_rate_decay_steps=TRAINING_STEPS,
      momentum=0.0,
      lars_coefficient=0.01,
  )

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Initial value:', sess.run(W)[0])
    for i in range(TRAINING_STEPS):
      if i % OUT_FREQ == 0:
        print('#{} Loss={:.4f}, W={:.4f}'.format(
            i, sess.run(loss), sess.run(W)[0]))
      sess.run(train_step)
    print('Predicted answer:', sess.run(W)[0])
  print('Correct answer:', answer)


if __name__ == '__main__':
  main()

