# coding=utf-8
# Copyright 2017 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""transformer (attention).

encoder: [Self-Attention, Feed-forward] x n
decoder: [Self-Attention, Source-Target-Attention, Feed-forward] x n
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports

from tensor2tensor.layers import common_layers
from tensor2tensor.layers import common_attention
from tensor2tensor.utils import t2t_model
from tensor2tensor.utils import registry
from tensor2tensor.utils import beam_search
from tensor2tensor.models import transformer

import tensorflow as tf
import numpy as np
import copy
import time
import six

from tensorflow.python.util import nest

from ..layers.common_layers import *
from ..layers.common_attention import * 

@registry.register_model
class TransformerFast(transformer.Transformer):
  """Fast decoding Transformer with Caching"""

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_self_attention_bias,
             hparams,
             cache=None):
    """Decode Transformer outputs from encoder representation.
    Args:
      decoder_input: inputs to bottom of the model.
          [batch_size, decoder_length, hidden_dim]
      encoder_output: Encoder representation.
          [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for
          encoder-decoder attention. [batch_size, input_length]
      decoder_self_attention_bias: Bias for self-attention.
      hparams: hyperparmeters for model.
      cache: dict, containing tensors which are the results of previous
          attentions, used for fast decoding.
    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer_decoder_fast(
        decoder_input,
        encoder_output,
        decoder_self_attention_bias,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache)

    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2)    

  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0,
                   eos_id=beam_search.EOS_ID):
    """Fast decoding.
    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.
    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.
    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams

    inputs = features["inputs"]
    batch_size = tf.shape(inputs)[0]
    target_modality = self._problem_hparams.target_modality
    if t2t_model.is_class_modality(target_modality):
      decode_length = 1
    else:
      decode_length = tf.shape(inputs)[1] + decode_length

    # TODO(llion): Clean up this reshaping logic.
    inputs = tf.expand_dims(inputs, axis=1)
    if len(inputs.shape) < 5:
      inputs = tf.expand_dims(inputs, axis=4)
    s = tf.shape(inputs)
    inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
    # _shard_features called to ensure that the variable names match
    inputs = self._shard_features({"inputs": inputs})["inputs"]
    input_modality = self._problem_hparams.input_modality["inputs"]
    with tf.variable_scope(input_modality.name):
      inputs = input_modality.bottom_sharded(inputs, dp)
    with tf.variable_scope("body"):
      encoder_output, encoder_decoder_attention_bias = dp(
          self.encode, inputs, features["target_space_id"], hparams)
    encoder_output = encoder_output[0]
    encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]

    if hparams.pos == "timing":
      timing_signal = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.
      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.
      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.
      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)[0]
      targets = common_layers.flatten4d3d(targets)

      # TODO(llion): Explain! Is this even needed?
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if hparams.pos == "timing":
        targets += timing_signal[:, i:i + 1]
      return targets

    decoder_self_attention_bias = (
        common_attention.attention_bias_lower_triangle(decode_length))
    if hparams.proximity_bias:
      decoder_self_attention_bias += common_attention.attention_bias_proximal(
          decode_length)

    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

      with tf.variable_scope("body"):
        body_outputs = dp(self.decode, targets, cache["encoder_output"],
                          cache["encoder_decoder_attention_bias"], bias,
                          hparams, cache)

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      return tf.squeeze(logits, axis=[1, 2, 3]), cache

    key_channels = hparams.attention_key_channels or hparams.hidden_size
    value_channels = hparams.attention_value_channels or hparams.hidden_size
    num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers

    # encoder-decoder attention keys and values.
    encdec_attn = {}
    with tf.variable_scope("body/decoder"):
      for layer in range(num_layers):
        layer_name = "layer_%d" % layer
        with tf.variable_scope(layer_name):
          with tf.variable_scope("encdec_attention/multihead_attention"):
            combined = common_layers.conv1d(
                        encoder_output,
                        key_channels + value_channels,
                        1,
                        padding="VALID",
                        name="kv_transform")
            k, v = tf.split(combined, [key_channels, value_channels], axis=2)
            encdec_attn[layer] = {"k": k, "v": v}


    src_step = tf.shape(encoder_output)[1]
    cache = {
        "layer_%d" % layer: {
            "k": tf.zeros([batch_size, 0, key_channels]),
            "v": tf.zeros([batch_size, 0, value_channels]),
            "k_encdec": encdec_attn[layer]["k"],
            "v_encdec": encdec_attn[layer]["v"],
        }
        for layer in range(num_layers)
    }

    # Set 2nd dim to None since it's not invariant in the tf.while_loop
    # Note: Tensor.set_shape() does not work here since it merges shape info.
    # TODO(llion); Find a more robust solution.
    # pylint: disable=protected-access
    for layer in cache:
      cache[layer]["k"]._shape = tf.TensorShape([None, None, key_channels])
      cache[layer]["v"]._shape = tf.TensorShape([None, None, value_channels])
    # pylint: enable=protected-access
    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    if beam_size > 1:  # Beam Search
      target_modality = (
          self._hparams.problems[self._problem_idx].target_modality)
      vocab_size = target_modality.top_dimensionality
      initial_ids = tf.zeros([batch_size], dtype=tf.int32)
      decoded_ids, scores = beam_search.beam_search(
          symbols_to_logits_fn, initial_ids, beam_size, decode_length,
          vocab_size, alpha, states=cache, stop_early=(top_beams == 1))

      if top_beams == 1:
        decoded_ids = decoded_ids[:, 0, 1:]
      else:
        decoded_ids = decoded_ids[:, :top_beams, 1:]
    else:  # Greedy

      def inner_loop(i, finished, next_id, decoded_ids, cache):
        logits, cache = symbols_to_logits_fn(next_id, i, cache)
        temperature = (0.0 if hparams.sampling_method == "argmax"
                       else hparams.sampling_temp)
        next_id = common_layers.sample_with_temperature(logits, temperature)
        finished |= tf.equal(next_id, eos_id)

        next_id = tf.expand_dims(next_id, axis=1)
        decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
        return i + 1, finished, next_id, decoded_ids, cache

      def is_not_finished(i, finished, *_):
        return (i < decode_length) & tf.logical_not(tf.reduce_all(finished))

      decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
      finished = tf.fill([batch_size], False)
      scores = None
      next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
      _, _, _, decoded_ids, _ = tf.while_loop(
          is_not_finished,
          inner_loop,
          [tf.constant(0), finished, next_id, decoded_ids, cache],
          shape_invariants=[
              tf.TensorShape([]),
              tf.TensorShape([None]),
              tf.TensorShape([None, None]),
              tf.TensorShape([None, None]),
              nest.map_structure(lambda t: tf.TensorShape(t.shape), cache),
          ])

    return decoded_ids, scores 

@registry.register_model
class TransformerFastAan(transformer.Transformer):
  """Fast decoding Transformer with Average Attention Layers (https://arxiv.org/abs/1805.00631)."""

  def decode(self,
             decoder_input,
             encoder_output,
             encoder_decoder_attention_bias,
             decoder_position_forward_mask,
             hparams,
             cache=None):
    """Decode Transformer outputs from encoder representation.
    Args:
      decoder_input: inputs to bottom of the model.
          [batch_size, decoder_length, hidden_dim]
      encoder_output: Encoder representation.
          [batch_size, input_length, hidden_dim]
      encoder_decoder_attention_bias: Bias and mask weights for
          encoder-decoder attention. [batch_size, input_length]
      decoder_position_forward_mask: mask Tensor for position-forward. [1, t, 1]
      hparams: hyperparmeters for model.
      cache: dict, containing tensors which are the results of previous
          attentions, used for fast decoding.
    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    decoder_input = tf.nn.dropout(decoder_input,
                                  1.0 - hparams.layer_prepostprocess_dropout)

    decoder_output = transformer_decoder_fast_aan(
        decoder_input,
        encoder_output,
        decoder_position_forward_mask,
        encoder_decoder_attention_bias,
        hparams,
        cache=cache)
    
    # Expand since t2t expects 4d tensors.
    return tf.expand_dims(decoder_output, axis=2)

  def model_fn_body(self, features):
    """Transformer main model_fn.
    Args:
      features: Map of features to the model. Should contain the following:
          "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
          "tragets": Target decoder outputs.
              [batch_size, decoder_length, hidden_dim]
          "target_space_id"
    Returns:
      Final decoder representation. [batch_size, decoder_length, hidden_dim]
    """
    hparams = self._hparams

    inputs = features.get("inputs")
    encoder_output, encoder_decoder_attention_bias = (None, None)
    if inputs is not None:
      target_space = features["target_space_id"]
      encoder_output, encoder_decoder_attention_bias = self.encode(
          inputs, target_space, hparams)

    targets = features["targets"]
    targets = common_layers.flatten4d3d(targets)

    decoder_input, decoder_position_forward_mask = transformer_fast_prepare_decoder(
        targets, hparams)

    return self.decode(decoder_input, encoder_output,
                       encoder_decoder_attention_bias,
                       decoder_position_forward_mask, hparams)
   
  def _fast_decode(self,
                   features,
                   decode_length,
                   beam_size=1,
                   top_beams=1,
                   alpha=1.0,
                   eos_id=beam_search.EOS_ID):
    """Fast decoding.
    Implements both greedy and beam search decoding, uses beam search iff
    beam_size > 1, otherwise beam search related arguments are ignored.
    Args:
      features: a map of string to model  features.
      decode_length: an integer.  How many additional timesteps to decode.
      beam_size: number of beams.
      top_beams: an integer. How many of the beams to return.
      alpha: Float that controls the length penalty. larger the alpha, stronger
        the preference for slonger translations.
    Returns:
       samples: an integer `Tensor`. Top samples from the beam search
    Raises:
      NotImplementedError: If there are multiple data shards.
    """
    if self._num_datashards != 1:
      raise NotImplementedError("Fast decoding only supports a single shard.")
    dp = self._data_parallelism
    hparams = self._hparams

    inputs = features["inputs"]
    batch_size = tf.shape(inputs)[0]
    target_modality = self._problem_hparams.target_modality
    if t2t_model.is_class_modality(target_modality):
      decode_length = 1
    else:
      decode_length = tf.shape(inputs)[1] + decode_length

    # TODO(llion): Clean up this reshaping logic.
    inputs = tf.expand_dims(inputs, axis=1)
    if len(inputs.shape) < 5:
      inputs = tf.expand_dims(inputs, axis=4)
    s = tf.shape(inputs)
    inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
    # _shard_features called to ensure that the variable names match
    inputs = self._shard_features({"inputs": inputs})["inputs"]
    input_modality = self._problem_hparams.input_modality["inputs"]
    with tf.variable_scope(input_modality.name):
      inputs = input_modality.bottom_sharded(inputs, dp)
    with tf.variable_scope("body"):
      encoder_output, encoder_decoder_attention_bias = dp(
          self.encode, inputs, features["target_space_id"], hparams)
    encoder_output = encoder_output[0]
    encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]

    if hparams.pos == "timing":
      timing_signal = common_attention.get_timing_signal_1d(
          decode_length + 1, hparams.hidden_size)

    def preprocess_targets(targets, i):
      """Performs preprocessing steps on the targets to prepare for the decoder.
      This includes:
        - Embedding the ids.
        - Flattening to 3D tensor.
        - Optionally adding timing signals.
      Args:
        targets: inputs ids to the decoder. [batch_size, 1]
        i: scalar, Step number of the decoding loop.
      Returns:
        Processed targets [batch_size, 1, hidden_dim]
      """
      # _shard_features called to ensure that the variable names match
      targets = self._shard_features({"targets": targets})["targets"]
      with tf.variable_scope(target_modality.name):
        targets = target_modality.targets_bottom_sharded(targets, dp)[0]
      targets = common_layers.flatten4d3d(targets)

      # TODO(llion): Explain! Is this even needed?
      targets = tf.cond(
          tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

      if hparams.pos == "timing":
        targets += timing_signal[:, i:i + 1]
      return targets


    def symbols_to_logits_fn(ids, i, cache):
      """Go from ids to logits for next symbol."""
      ids = ids[:, -1:]
      targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
      targets = preprocess_targets(targets, i)

      mask = tf.ones([1, 1, 1]) / (tf.to_float(i) + 1.)

      with tf.variable_scope("body"):
        body_outputs = dp(self.decode, targets, cache["encoder_output"],
                          cache["encoder_decoder_attention_bias"], mask,
                          hparams, cache)

      with tf.variable_scope(target_modality.name):
        logits = target_modality.top_sharded(body_outputs, None, dp)[0]

      return tf.squeeze(logits, axis=[1, 2, 3]), cache

    key_channels = hparams.attention_key_channels or hparams.hidden_size
    value_channels = hparams.attention_value_channels or hparams.hidden_size
    num_layers = hparams.num_decoder_layers or hparams.num_hidden_layers

    # encoder-decoder attention keys and values.
    encdec_attn = {}
    with tf.variable_scope("body/decoder"):
      for layer in range(num_layers):
        layer_name = "layer_%d" % layer
        with tf.variable_scope(layer_name):
          with tf.variable_scope("encdec_attention/multihead_attention"):
            combined = common_layers.conv1d(
                        encoder_output,
                        key_channels + value_channels,
                        1,
                        padding="VALID",
                        name="kv_transform")
            k, v = tf.split(combined, [key_channels, value_channels], axis=2)
            encdec_attn[layer] = {"k": k, "v": v}


    src_step = tf.shape(encoder_output)[1]
    cache = {
        "layer_%d" % layer: {
            "k_encdec": encdec_attn[layer]["k"],
            "v_encdec": encdec_attn[layer]["v"],
            "given_inputs": tf.zeros([batch_size, 1, hparams.hidden_size])
        }
        for layer in range(num_layers)
    }

    # Set 2nd dim to None since it's not invariant in the tf.while_loop
    # Note: Tensor.set_shape() does not work here since it merges shape info.
    # TODO(llion); Find a more robust solution.
    # pylint: disable=protected-access
    for layer in cache:
      cache[layer]["given_inputs"]._shape = tf.TensorShape([None, None, hparams.hidden_size])
    # pylint: enable=protected-access
    cache["encoder_output"] = encoder_output
    cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias

    if beam_size > 1:  # Beam Search
      target_modality = (
          self._hparams.problems[self._problem_idx].target_modality)
      vocab_size = target_modality.top_dimensionality
      initial_ids = tf.zeros([batch_size], dtype=tf.int32)
      decoded_ids, scores = beam_search.beam_search(
          symbols_to_logits_fn, initial_ids, beam_size, decode_length,
          vocab_size, alpha, states=cache, stop_early=(top_beams == 1))

      if top_beams == 1:
        decoded_ids = decoded_ids[:, 0, 1:]
      else:
        decoded_ids = decoded_ids[:, :top_beams, 1:]
    else:  # Greedy

      def inner_loop(i, finished, next_id, decoded_ids, cache):
        logits, cache = symbols_to_logits_fn(next_id, i, cache)
        temperature = (0.0 if hparams.sampling_method == "argmax"
                       else hparams.sampling_temp)
        next_id = common_layers.sample_with_temperature(logits, temperature)
        finished |= tf.equal(next_id, eos_id)

        next_id = tf.expand_dims(next_id, axis=1)
        decoded_ids = tf.concat([decoded_ids, next_id], axis=1)
        return i + 1, finished, next_id, decoded_ids, cache

      def is_not_finished(i, finished, *_):
        return (i < decode_length) & tf.logical_not(tf.reduce_all(finished))

      decoded_ids = tf.zeros([batch_size, 0], dtype=tf.int64)
      finished = tf.fill([batch_size], False)
      scores = None
      next_id = tf.zeros([batch_size, 1], dtype=tf.int64)
      _, _, _, decoded_ids, _ = tf.while_loop(
          is_not_finished,
          inner_loop,
          [tf.constant(0), finished, next_id, decoded_ids, cache],
          shape_invariants=[
              tf.TensorShape([]),
              tf.TensorShape([None]),
              tf.TensorShape([None, None]),
              tf.TensorShape([None, None]),
              nest.map_structure(lambda t: tf.TensorShape(t.shape), cache),
          ])

    return decoded_ids, scores 


def transformer_decoder_fast(decoder_input,
                        encoder_output,
                        decoder_self_attention_bias,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        name="decoder"):
  """A stack of transformer layers.
  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_position_forward_mask: mask Tensor for position-forward / shape: [1, t, 1]
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    name: a string
  Returns:
    y: a Tensors
  """
  x = decoder_input
  with tf.variable_scope(name):
    for layer in range(hparams.num_decoder_layers or
                        hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("self_attention"):
          y = common_attention.multihead_attention(
              common_layers.layer_preprocess(x, hparams),
              None,
              decoder_self_attention_bias,
              hparams.attention_key_channels or hparams.hidden_size,
              hparams.attention_value_channels or hparams.hidden_size,
              hparams.hidden_size,
              hparams.num_heads,
              hparams.attention_dropout,
              attention_type=hparams.self_attention_type,
              max_relative_position=hparams.max_relative_position,
              cache=layer_cache)
          x = common_layers.layer_postprocess(x, y, hparams)
        if encoder_output is not None:
          with tf.variable_scope("encdec_attention"):
            y = multihead_attention(
                common_layers.layer_preprocess(
                    x, hparams), encoder_output, encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size, hparams.num_heads,
                hparams.attention_dropout,
                cache=layer_cache)
            x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer.transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams), hparams)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)


def transformer_decoder_fast_aan(decoder_input,
                        encoder_output,
                        decoder_position_forward_mask,
                        encoder_decoder_attention_bias,
                        hparams,
                        cache=None,
                        name="decoder"):
  """A stack of transformer layers.
  Args:
    decoder_input: a Tensor
    encoder_output: a Tensor
    decoder_position_forward_mask: mask Tensor for position-forward / shape: [1, t, 1]
    encoder_decoder_attention_bias: bias Tensor for encoder-decoder attention
      (see common_attention.attention_bias())
    hparams: hyperparameters for model
    cache: dict, containing tensors which are the results of previous
        attentions, used for fast decoding.
    name: a string
  Returns:
    y: a Tensors
  """
  x = decoder_input
  with tf.variable_scope(name):
    for layer in range(hparams.num_decoder_layers or
                        hparams.num_hidden_layers):
      layer_name = "layer_%d" % layer
      layer_cache = cache[layer_name] if cache is not None else None
      with tf.variable_scope(layer_name):
        with tf.variable_scope("position_forward"):
          if layer_cache:
            given_inputs_new = layer_cache['given_inputs'] + x
            x_fwd = given_inputs_new * decoder_position_forward_mask
            layer_cache['given_inputs'] = given_inputs_new + x
          else:
            x_fwd = tf.cumsum(x, axis=1) * decoder_position_forward_mask
          # FFN activation
          y = transformer.transformer_ffn_layer(
              common_layers.layer_preprocess(x_fwd, hparams), hparams)

          # Gating layer
          z = tf.layers.dense(tf.concat([x, y], axis=-1), hparams.hidden_size*2, name="z_project")
          i, f = tf.split(z, 2, axis=-1)
          y = tf.sigmoid(i) * x + tf.sigmoid(f) * y
          x = common_layers.layer_postprocess(x, y, hparams)

        if encoder_output is not None:
          with tf.variable_scope("encdec_attention"):
            y = multihead_attention(
                common_layers.layer_preprocess(
                    x, hparams), encoder_output, encoder_decoder_attention_bias,
                hparams.attention_key_channels or hparams.hidden_size,
                hparams.attention_value_channels or hparams.hidden_size,
                hparams.hidden_size, hparams.num_heads,
                hparams.attention_dropout,
                cache=layer_cache)
            x = common_layers.layer_postprocess(x, y, hparams)
        with tf.variable_scope("ffn"):
          y = transformer.transformer_ffn_layer(
              common_layers.layer_preprocess(x, hparams), hparams)
          x = common_layers.layer_postprocess(x, y, hparams)
    # if normalization is done in layer_preprocess, then it shuold also be done
    # on the output, since the output can grow very large, being the sum of
    # a whole stack of unnormalized layer outputs.
    return common_layers.layer_preprocess(x, hparams)

def compute_q(query_antecedent, total_key_depth, q_filter_width=1, q_padding="VALID"):
  """Computes query.
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    total_key_depth: an integer
    q_filter_width: An integer specifying how wide you want the query to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
  Returns:
    q: [batch, length, depth] tensors
  """
  q = common_layers.conv1d(
      query_antecedent,
      total_key_depth,
      q_filter_width,
      padding=q_padding,
      name="q_transform")
  return q

def multihead_attention(query_antecedent,
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
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name=None,
                        **kwargs):
  """Multihead scaled-dot-product attention with input/output transformations.
  Args:
    query_antecedent: a Tensor with shape [batch, length_q, channels]
    memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
    bias: bias Tensor (see attention_bias())
    total_key_depth: an integer
    total_value_depth: an integer
    output_depth: an integer
    num_heads: an integer dividing total_key_depth and total_value_depth
    dropout_rate: a floating point number
    max_relative_position: Maximum distance between inputs to generate
                           unique relation embeddings for. Only relevant
                           when using "dot_product_relative" attention.
    image_shapes: optional tuple of integer scalars.
                  see comments for attention_image_summary()
    attention_type: a string, either "dot_product", "dot_product_relative",
                    "local_mask_right", "local_unmasked", "masked_dilated_1d",
                    "unmasked_dilated_1d" or any attention function with the
                    signature (query, key, value, **kwargs)
    block_length: an integer - relevant for "local_mask_right"
    block_width: an integer - relevant for "local_unmasked"
    q_filter_width: An integer specifying how wide you want the query to be.
    kv_filter_width: An integer specifying how wide you want the keys and values
                     to be.
    q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
               kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
               no padding.
    cache: dict containing Tensors which are the results of previous
           attentions, used for fast decoding. Expects the dict to contrain two
           keys ('k' and 'v'), for the initial call the values for these keys
           should be empty Tensors of the appropriate shape.
               'k' [batch_size, 0, key_channels]
               'v' [batch_size, 0, value_channels]
    gap_size: Integer option for dilated attention to indicate spacing between
              memory blocks.
    num_memory_blocks: Integer option to indicate how many memory blocks to look
                       at.
    name: an optional string
    **kwargs (dict): Parameters for the attention function
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

    if cache is None:
      q, k, v = common_attention.compute_qkv(query_antecedent, memory_antecedent, total_key_depth,
                            total_value_depth, q_filter_width, kv_filter_width,
                            q_padding, kv_padding)
    else:
      q = compute_q(query_antecedent, total_key_depth, q_filter_width, q_padding)
      k, v = cache['k_encdec'], cache['v_encdec']

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


def transformer_fast_prepare_decoder(targets, hparams):
  """Prepare one shard of the model for the decoder.
  Args:
    targets: a Tensor.
    hparams: run hyperparameters
  Returns:
    decoder_input: a Tensor, bottom of decoder stack
    decoder_position_forward_mask: mask Tensor for position-forward. [1, t, 1]
  """
  length = tf.shape(targets)[1]
  decoder_position_forward_mask = 1. / tf.expand_dims(tf.expand_dims(tf.to_float(tf.range(length)) + 1., 0), -1) # [1, t, 1]

  decoder_input = common_layers.shift_right_3d(targets)
  if hparams.pos == "timing":
    decoder_input = common_attention.add_timing_signal_1d(decoder_input)
  return (decoder_input, decoder_position_forward_mask)
