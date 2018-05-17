# coding=utf-8

"""Utilities for attention."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention

import tensorflow as tf


def attention_bias_sar(length, n_leap):
  """Create an bias tensor to be added to semi-autoregressive attention logits."""
  def body(i, x):
    no_attn = tf.ones([i, n_leap])
    ok_attn = tf.zeros([length-i, n_leap])
    local_attn = tf.concat([no_attn, ok_attn], 0)
    x = tf.concat([x, local_attn], 1)
    return [tf.add(i, n_leap), x]
    
  i = tf.constant(0)
  x = tf.zeros([length, 0])
  _, bias = tf.while_loop(lambda i, x: tf.less(i, length), body, [i, x], 
                       shape_invariants=[i.get_shape(), tf.TensorShape([None, None])],
                       name='attn_bias_loop')
  bias = bias * -1e9
  return tf.reshape(bias, [1, 1, length, length])


def attention_bias_center(attn_bias, w_size, value=10.):
  bias_mask = tf.cast(tf.equal(attn_bias[0,0], 0), attn_bias.dtype)
  centered_bias = tf.matrix_band_part(bias_mask, w_size-1, w_size-1) * value
  centered_bias = tf.expand_dims(tf.expand_dims(centered_bias, 0), 0)
  return centered_bias


def compute_qkv_pos(query_antecedent, memory_antecedent, total_key_depth,
                total_value_depth, qkv_padding="VALID"):
  """Computes query, key and value.
  Returns:
    q, k, v : [batch, length, depth] tensors
  """
  if memory_antecedent is None:
    # self attention
    combined = common_layers.conv1d(
        query_antecedent,
        total_key_depth * 2 + total_value_depth,
        1,
        padding=qkv_padding,
        name="qkv_transform")
    q, k, v = tf.split(
        combined, [total_key_depth, total_key_depth, total_value_depth],
        axis=2)
    return q, k, v

  # encoder-decoder attention
  q = common_layers.conv1d(
      query_antecedent, total_key_depth, 1, padding=qkv_padding,
      name="q_transform")
  k = common_layers.conv1d(
      query_antecedent, total_key_depth, 1, padding=qkv_padding,
      name="k_transform")
  v = common_layers.conv1d(
      memory_antecedent, total_value_depth, 1, padding=qkv_padding,
      name="v_transform")

  return q, k, v


def multihead_attention_pos(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        max_relative_position=None,
                        image_shapes=None,
                        attention_type="dot_product",
                        block_length=128,
                        block_width=128,
                        qkv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name=None,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.
  Caching:
    WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
    the caching assumes that the bias contains future masking.
    The caching works by saving all the previous key and value values so that
    you are able to send just the last query location to this attention
    function. I.e. if the cache dict is provided it assumes the query is of the
    shape [batch_size, 1, hiddem_dim] rather than the full memory.
  Returns:
    The result of the attention transformation. The output shape is
        [batch_size, length_q, hidden_dim]
    unless the cache dict is provided in which case only the last memory
    position is calculated and the output shape is [batch_size, 1, hidden_dim]
    Optionaly returns an additional loss parameters (ex: load balance loss for
    the experts) returned by the attention_type function.
  Raises:
    ValueError: if the key depth or value depth are not divisible by the
      number of attention heads.
  """
  if total_key_depth % num_heads != 0:
    raise ValueError("Key depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_key_depth, num_heads))
  if total_value_depth % num_heads != 0:
    raise ValueError("Value depth (%d) must be divisible by the number of "
                     "attention heads (%d)." % (total_value_depth, num_heads))
  with tf.variable_scope(
      name,
      default_name="multihead_attention",
      values=[query_antecedent, memory_antecedent]):
    q, k, v = compute_qkv_pos(query_antecedent, memory_antecedent, total_key_depth,
                          total_value_depth, qkv_padding)

    if cache is not None:
      if attention_type != "dot_product":
        raise NotImplementedError(
            "Caching is not guaranteed to work with attention types other than"
            " dot_product.")
      if bias is None:
        raise ValueError("Bias required for caching. See function docstring "
                         "for details.")
      k = cache["k"] = tf.concat([cache["k"], k], axis=1)
      v = cache["v"] = tf.concat([cache["v"], v], axis=1)

    q = common_attention.split_heads(q, num_heads)
    k = common_attention.split_heads(k, num_heads)
    v = common_attention.split_heads(v, num_heads)
    key_depth_per_head = total_key_depth // num_heads
    q *= key_depth_per_head**-0.5

    additional_returned_value = None
    if callable(attention_type):  # Generic way to extend multihead_attention
      x = attention_type(q, k, v, **kwargs)
      if isinstance(x, tuple):
        x, additional_returned_value = x  # Unpack
    elif attention_type == "dot_product":
      x = common_attention.dot_product_attention(q, k, v, bias, dropout_rate, image_shapes)
    elif attention_type == "dot_product_relative":
      x = common_attention.dot_product_attention_relative(q, k, v, bias, max_relative_position,
                                         dropout_rate, image_shapes)
    elif attention_type == "local_mask_right":
      x = common_attention.masked_local_attention_1d(q, k, v, block_length=block_length)
    elif attention_type == "local_unmasked":
      x = common_attention.local_attention_1d(
          q, k, v, block_length=block_length, filter_width=block_width)
    elif attention_type == "masked_dilated_1d":
      x = common_attention.masked_dilated_self_attention_1d(q, k, v, block_length,
                                           block_width,
                                           gap_size,
                                           num_memory_blocks)
    else:
      assert attention_type == "unmasked_dilated_1d"
      x = common_attention.dilated_self_attention_1d(q, k, v, block_length,
                                    block_width,
                                    gap_size,
                                    num_memory_blocks)
    x = common_attention.combine_heads(x)
    x = common_layers.conv1d(x, output_depth, 1, name="output_transform")
    if additional_returned_value is not None:
      return x, additional_returned_value
    return x

