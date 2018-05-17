# coding=utf-8

"""Layers common to multiple models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import tensorflow as tf


def layer_norm_stats(x, epsilon=1e-6):
  """Layer norm raw computation."""
  mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
  variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)

  shift = mean #-mean * tf.rsqrt(variance + epsilon)
  log_scale = 0.5 * tf.log(variance + epsilon) #-0.5 * tf.log(variance + epsilon)
  return shift, log_scale


def layer_norm(x, epsilon=1e-6, return_stats=False):
  """Layer norm raw computation."""
  mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
  variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
  norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
  return norm_x


def mod_seq_len(x, m):
  """expand sequence length to become it as a modulus of m."""
  x_dim = tf.shape(x)
  batch_size, max_seq_len, hidden_dim = x_dim[0], x_dim[1], x_dim[2]
    
  # calculate how much length is needed to make it as a modulus
  dummy_len = (m - (max_seq_len % m)) % m
  dummy = tf.zeros([batch_size, dummy_len, hidden_dim])
  expanded_x = tf.concat([x, dummy], 1)
  return expanded_x


def shift_right_3d_multiple(x, steps=1, pad_value=0, remove=True, trainable=False, name="shift_3d"):
  """Shift the second dimension of x right by n steps."""
  with tf.variable_scope(name):
    batch_size = tf.shape(x)[0]
    hidden_dim = x.get_shape()[2]
    pad_tensor = tf.zeros([1, steps, hidden_dim]) + pad_value
    if trainable:
      pad_tensor = tf.get_variable("pad_tensor", initializer=pad_tensor)
    pad_tensor = tf.tile(pad_tensor, [batch_size, 1, 1])
    shifted_targets = tf.concat([pad_tensor, x], axis=1)
      
    if remove:
      shifted_targets = shifted_targets[:, :-steps, :]
  return shifted_targets

def causal_dense_relu_statistics(x, num_hidden_layers, output_units):
  for i in range(num_hidden_layers):
    x = dense_wn(
        inputs=x,
        units=output_units,
        name='causal_dense_wn_%d' % i,
        activation=tf.nn.relu,
        causal=True)
  shift = dense_wn(
      inputs=x,
      units=output_units,
      name='causal_dense_wn_shift',
      activation=None,
      causal=True)
  log_scale = dense_wn(
      inputs=x,
      units=output_units,
      name='causal_dense_wn_log_scale',
      activation=None,
      causal=True)
      
  scale = tf.get_variable("rescaling_scale", [output_units,], initializer=tf.constant_initializer(0.))
  scale_shift = tf.get_variable("scale_shift", [output_units,], initializer=tf.constant_initializer(0.))
  log_rescale = scale * tf.tanh(log_scale) + scale_shift
  return shift, log_rescale

def dense_relu_statistics(x, hidden_layers, output_units):
  for i, hidden_units in enumerate(hidden_layers):
    x = dense_wn(
        inputs=x,
        units=hidden_units,
        name='dense_wn_%d' % i,
        activation=tf.nn.relu)
  x = dense_wn(
      inputs=x,
      units=2 * output_units,
      name='dense_wn_stats',
      activation=None)
  
  shift, log_scale = tf.split(x, 2, axis=-1)
  scale = tf.get_variable("rescaling_scale", [output_units,], initializer=tf.constant_initializer(0.))
  scale_shift = tf.get_variable("scale_shift", [output_units,], initializer=tf.constant_initializer(0.))
  log_rescale = scale * tf.tanh(log_scale) + scale_shift
  return shift, log_rescale


def dense_wn(
    inputs, 
    units,
    name ='dense_wn',
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    causal=False):
  """Functional interface for the weight normaized densely-connected layer.
  """
  with tf.variable_scope(name):
    input_shape = inputs.get_shape().as_list()
    kernel = tf.get_variable('kernel',
      shape=[input_shape[-1], units],
      initializer=kernel_initializer,
      regularizer=kernel_regularizer,
      constraint=kernel_constraint,
      trainable=True)
    gain = tf.get_variable('gain',
      shape=[1, units],
      initializer=tf.constant_initializer(1.),
      regularizer=kernel_regularizer,
      constraint=kernel_constraint,
      trainable=True)
    if use_bias:
      bias = tf.get_variable('bias',
        shape=[units,],
        initializer=bias_initializer,
        regularizer=bias_regularizer,
        constraint=bias_constraint,
        trainable=True)
    if not causal:
      inv_norm = tf.rsqrt(tf.reduce_sum(tf.square(kernel), 0, keep_dims=True))
    else:
      kernel_mask = tf.ones_like(kernel)
      kernel_mask = tf.matrix_band_part(kernel_mask, 0, -1) - tf.matrix_band_part(kernel_mask, 0, 0)
      kernel *= kernel_mask
      inv_norm = tf.concat([tf.zeros([1,1]), tf.rsqrt(tf.reduce_sum(tf.square(kernel[:,1:]), 0, keep_dims=True))], 1)
      
    normalized_kernel = gain * kernel * inv_norm
    outputs = tf.tensordot(inputs, normalized_kernel, [[len(input_shape) - 1], [0]])
    if use_bias:
      outputs += bias
    if activation:
      outputs = activation(outputs)
  return outputs


def embedding_mask(emb):
  emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
  return tf.expand_dims(tf.to_float(tf.not_equal(emb_sum, 0.0)), -1)


