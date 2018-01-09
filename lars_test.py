#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import numpy.testing as npt
import lars


# TODO use GATE_OP for testing (minimize(..., gate_gradients=GATE_OP)??

def run_lars(w, loss, steps, b=None, eval_debug_fn=False, lars_coefficient=0.01,
             **kwargs):
  train_step, debug_fn = lars.createLarsMinimizer(
      loss=loss,
      initial_learning_rate=1.0,
      learning_rate_decay_steps=steps,
      momentum=0.0,
      lars_coefficient=lars_coefficient,
      **kwargs
  )
  debug_results = []
  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(steps):
      if eval_debug_fn:
        debug_results.append(debug_fn(sess))
      # TODO: make an option to output debug info: each output_freq (new
      #     param) iterations log current weights norm, current gradient norm,
      #     current objective
      # print(sess.run(w))
      sess.run(train_step)
    w_pred = sess.run(w)
    b_pred = None if b is None else sess.run(b)

    if eval_debug_fn:
      debug_results.append(debug_fn(sess))

  return w_pred, b_pred, debug_results


# If b_true (true bias value) defined, then the bias variable is created.
def linear_func(w_true, size, b_true=None, noise=0):
  # works in 2d only
  assert(len(size) == 2)

  X_np = np.random.uniform(-100, 100, size)
  y_np = X_np.dot(w_true)
  if noise != 0:
    y_np += np.random.normal(0, noise, y_np.shape)
    w_true = np.linalg.inv(X_np.T.dot(X_np)).dot(X_np.T).dot(y_np)

  X = tf.constant(X_np, dtype=tf.float32)
  w = tf.Variable(tf.random_normal(w_true.shape))
  if b_true is not None:
    y_np += b_true
    b = tf.Variable(tf.zeros(b_true.shape))
    predicted_y = tf.matmul(X, w) + b
  else:
    b = None
    predicted_y = tf.matmul(X, w)

  y = tf.constant(y_np, dtype=tf.float32)
  loss = tf.losses.mean_squared_error(y, predicted_y)
  return loss, w, w_true, b


class TestConvergenceLARS(tf.test.TestCase):
  # TODO: make some smaller tests, checking iterations?
  # TODO: test different values of lars_coefficient?
  # TODO: make momentum and weight decay tests?
  # TODO: test for a small neural network?

  def test_decaying_learning_rate(self):
    loss, w, _, __ = linear_func(np.array([[123.0]]), (10, 1))
    _, __, debug_results = run_lars(w, loss, 5, eval_debug_fn=True)
    global_steps = [x['global_step'] for x in debug_results]
    learning_rates = [x['learning_rate'] for x in debug_results]

    self.assertListEqual([0, 1, 2, 3, 4, 5], global_steps)
    self.assertArrayNear([1.0, 0.64, 0.36, 0.16, 0.04, 0.0], learning_rates,
                         err=0.0000001)

  def test_learning_rate_does_not_decay_if_use_decay_is_False(self):
    loss, w, _, __ = linear_func(np.array([[123.0]]), (10, 1))
    _, __, debug_results = run_lars(w, loss, 5, eval_debug_fn=True, use_decay=False)
    global_steps = [x['global_step'] for x in debug_results]
    learning_rates = [x['learning_rate'] for x in debug_results]

    self.assertListEqual([None] * 6, global_steps)
    self.assertListEqual([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], learning_rates)

  def test_decay_learning_rate_with_multiple_vars(self):
    # Use w and bias so multiple vars.
    loss, w, _, __ = linear_func(np.array([[123.0]]), (10, 1),
                                 b_true=np.array([10.0]))
    _, __, debug_results = run_lars(w, loss, 5, eval_debug_fn=True)
    global_steps = [x['global_step'] for x in debug_results]
    learning_rates = [x['learning_rate'] for x in debug_results]

    self.assertListEqual([0, 1, 2, 3, 4, 5], global_steps)
    self.assertArrayNear([1.0, 0.64, 0.36, 0.16, 0.04, 0.0], learning_rates,
                         err=0.0000001)

  def test_linear_big(self):
    loss, w, w_true, _ = linear_func(np.array([[123.0]]), (10, 1))
    w_pred, _, __ = run_lars(w, loss, 10000)
    npt.assert_almost_equal(w_pred, w_true, decimal=5)

  def test_linear_big_with_bias(self):
    b_true = np.array([10.0])
    # Use a smaller weight so converge faster. And thus test runs faster.
    loss, w, w_true, b = linear_func(np.array([[12.0]]), (10, 1), b_true=b_true)
    w_pred, b_pred, _ = run_lars(w, loss, 10000, b=b)
    npt.assert_almost_equal(w_pred, w_true, decimal=5)
    npt.assert_almost_equal(b_pred, b_true, decimal=4)

  def test_linear_highdim(self):
    loss, w, w_true, _ = linear_func(np.array([[123.0, -200, 5]]), (50, 1))
    w_pred, _, __ = run_lars(w, loss, 10000)
    npt.assert_almost_equal(w_pred, w_true, decimal=5)

  def test_linear_highdim_noise(self):
    loss, w, w_true, _ = linear_func(
        np.array([[123.0, -200, 5]]), (50, 1), noise=1)
    w_pred, _, __ = run_lars(w, loss, 10000)
    npt.assert_almost_equal(w_pred, w_true, decimal=5)

 # # TODO: find the right params to pass this test
 # def test_linear_small(self):
 #   loss, w, w_true, _ = linear_func(np.array([[0.0001]]), (10, 1))
 #   w_pred, _, __ = run_lars(w, loss, 10000, lars_coefficient=0.001)
 #   npt.assert_almost_equal(w_pred, w_true)


if __name__ == '__main__':
  tf.test.main()

