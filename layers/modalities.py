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

"""Modalities define the bottom and top of the model (not the body)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import registry
from tensor2tensor.layers import modalities
from tensor2tensor.layers import common_layers
import tensorflow as tf

@registry.register_symbol_modality("tgtemb")
class TargetEmbShareSymbolModality(modalities.SymbolModality):
  """SymbolModality that shares Target Embedding."""

  def targets_bottom(self, x):
    try:
      return self.bottom_simple(x, "shared", reuse=True)
    except ValueError:
      # perhaps there were no inputs, and this is a new variable.
      return self.bottom_simple(x, "shared", reuse=None)

  def top(self, body_output, _):
    """Generate logits.
    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_input_depth]
    Returns:
      logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
    """
    scope_name = "shared"
    reuse = True
    if self._model_hparams.symbol_modality_skip_top:
      return tf.expand_dims(body_output, 3)
    with tf.variable_scope(scope_name, reuse=reuse):
      var = self._get_weights()
      if (self._model_hparams.factored_logits and
          self._model_hparams.mode == tf.estimator.ModeKeys.TRAIN):
        # insert channels dimension
        body_output = tf.expand_dims(body_output, 3)
        logits = common_layers.FactoredTensor(body_output, var)
      else:
        shape = tf.shape(body_output)[:-1]
        body_output = tf.reshape(body_output, [-1, self._body_input_depth])
        logits = tf.matmul(body_output, var, transpose_b=True)
        logits = tf.reshape(
            logits, tf.concat([shape, [1, self._vocab_size]], 0))
      return logits

@registry.register_symbol_modality("mos")
class MixtureOfSoftmaxSymbolModality(modalities.SymbolModality):
  """SymbolModality that uses Mixture of Softmax."""

  def top(self, body_output, _):
    """Generate logits.
    Args:
      body_output: A Tensor with shape [batch, p0, p1, body_input_depth]
    Returns:
      logits: A Tensor with shape  [batch, p0, p1, ?, vocab_size].
    """ 
    if self._model_hparams.shared_embedding_and_softmax_weights:
      scope_name = "shared"
      reuse = True
    else:
      scope_name = "softmax"
      reuse = False
    if self._model_hparams.symbol_modality_skip_top:
      return tf.expand_dims(body_output, 3)
    with tf.variable_scope(scope_name, reuse=reuse):
      var = self._get_weights()
      shape = tf.shape(body_output)[:-1]
      body_output = tf.reshape(body_output, [-1, self._body_input_depth]) # [b * l, h]
                                 
    with tf.variable_scope(scope_name + '/mos'):
      latent = tf.layers.dense(body_output, 
                               self._model_hparams.n_experts * self._body_input_depth, 
                               activation=tf.nn.tanh) # [b * l, e * h] 
      latent = tf.nn.dropout(latent, 1.0 - self._model_hparams.layer_prepostprocess_dropout)
      
      # [b * l * e, h] x [h, v] = [b * l * e, v]
      logits = tf.matmul(tf.reshape(latent, [-1, self._body_input_depth]), var, transpose_b=True)
        
      prior_logit = tf.layers.dense(body_output, self._model_hparams.n_experts, use_bias=False) # [b * l, e]
      prior = tf.nn.softmax(prior_logit)
      
      prob = tf.reshape(tf.nn.softmax(logits), 
                        [-1, self._model_hparams.n_experts, self._vocab_size]) # [b * l, e, v]
      prob = tf.reduce_sum(prob * tf.expand_dims(prior, 2), 1) # [b * l, e, v] * [b * l, e, 1] -reduce-> [b * l, v]
    
      model_output = tf.log(prob + 1e-8)
      model_output = tf.reshape(model_output, tf.concat([shape, [1, self._vocab_size]], 0)) # [b, l, 1, v]
      return model_output
    