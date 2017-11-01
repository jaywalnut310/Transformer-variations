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
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

import tensorflow as tf


@registry.register_model
class TransformerChrawr(transformer.Transformer):
  """Transformer with Character-Aware Embedding."""

  def encode(self, inputs, target_space, hparams):
    """Encode transformer inputs.

    Args:
      inputs: Transformer inputs [batch_size, input_length, hidden_dim]
      target_space: scalar, target space ID.
      hparams: hyperparmeters for model.

    Returns:
      Tuple of:
          encoder_output: Encoder representation.
              [batch_size, input_length, hidden_dim]
          encoder_decoder_attention_bias: Bias and mask weights for
              encodre-decoder attention. [batch_size, input_length]
    """
    inputs = common_layers.flatten4d3d(inputs)
    
    ### Character-Aware Embedding ###
    inputs = tf.layers.conv1d(inputs, hparams.reduced_input_size, 1, 1, 'same', name="reduced_embedding")
    inputs = tdnn(inputs, hparams.chr_kernels, hparams.chr_kernel_features, hparams.chr_maxpool_size)
    inputs = highway(inputs, inputs.get_shape()[-1], hparams)
    inputs = tf.layers.conv1d(inputs, hparams.hidden_size, 1, 1, 'same', name="rescaled_embedding")

    encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
        transformer.transformer_prepare_encoder(inputs, target_space, hparams))

    encoder_input = tf.nn.dropout(
        encoder_input, 1.0 - hparams.layer_prepostprocess_dropout)

    encoder_output = transformer.transformer_encoder(
        encoder_input,
        self_attention_bias,
        hparams)

    return encoder_output, encoder_decoder_attention_bias

def highway(inputs, size, hparams, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """

    with tf.variable_scope(scope):
        for idx in range(hparams.num_highway_layers):
            t = tf.layers.conv1d(inputs, size, 1, 1, 'same', bias_initializer=tf.constant_initializer(bias), activation=tf.nn.sigmoid, name='highway_lin_%d' % idx)
            g = tf.layers.conv1d(inputs, size, 1, 1, 'same', activation=tf.nn.relu, name='highway_gate_%d' % idx)

            output = t * g + (1. - t) * inputs
            inputs = output

    return output


def tdnn(inputs, kernels, kernel_features, maxpool_size, scope='TDNN'):
    '''
    :inputs:           input float tensor of shape [(batch_size) x time_step x embed_size]
    :kernels:         array of kernel sizes
    :kernel_features: array of kernel feature sizes (parallel to kernels)
    '''
    assert len(kernels) == len(kernel_features), 'Kernel and Features must have the same size'

    layers = []
    with tf.variable_scope(scope):
        for kernel_size, kernel_feature_size in zip(kernels, kernel_features):

            # [batch_size x time_step x kernel_feature_size]
            conv = tf.layers.conv1d(inputs, kernel_feature_size, kernel_size, 1, 'same', activation=tf.nn.relu, name="kernel_%d" % kernel_size)

            # [batch_size x modified_time_step x kernel_feature_size]
            pool = tf.layers.max_pooling1d(tf.nn.relu(conv), maxpool_size, maxpool_size, 'same')
            
            layers.append(pool)

        if len(kernels) > 1:
            output = tf.concat(layers, 2)
        else:
            output = layers[0]

    return output # [batch_size x modified_time_step x hidden_dim]

@registry.register_hparams
def transformer_chrawr_base():
  """Base hparams for Transformer with Character Aware Embedding."""
  hparams = transformer.transformer_base()
  hparams.num_highway_layers = 3
  hparams.reduced_input_size = 128
  hparams.hidden_size = 512
  hparams.chr_kernels = [1,2,3,4,5,6,7,8]
  hparams.chr_kernel_features = [50,100,150,200,200,200,200,200]
  hparams.chr_maxpool_size = 5
  return hparams

@registry.register_hparams
def transformer_chrawr_big():
  """HParams for transfomer_chrawr big model on WMT."""
  hparams = transformer_chrawr_base()
  hparams.hidden_size = 1024
  hparams.chr_kernels = [1,2,3,4,5,6,7,8]
  hparams.chr_kernel_features = [80,80,112,112,160,160,160,160]
  hparams.filter_size = 4096
  hparams.num_heads = 16
  hparams.layer_prepostprocess_dropout = 0.3
  return hparams


@registry.register_hparams
def transformer_chrawr_big_single_gpu():
  """HParams for transformer_chrawr big model for single gpu."""
  hparams = transformer.transformer_chrawr_big()
  hparams.layer_prepostprocess_dropout = 0.1
  hparams.learning_rate_warmup_steps = 16000
  hparams.optimizer_adam_beta2 = 0.998
  return hparams


@registry.register_hparams
def transformer_chrawr_base_single_gpu():
  """HParams for transformer_chrawr base model for single gpu."""
  hparams = transformer_chrawr_base()
  hparams.batch_size = 2048
  hparams.learning_rate_warmup_steps = 16000
  return hparams

@registry.register_hparams
def transformer_chrawr_ko_single_gpu():
  """HParams for transformer_chrawr base model for single gpu."""
  hparams = transformer_chrawr_base_single_gpu()
  hparams.chr_kernels = [1,2,3,4,5,6,7,8]
  hparams.chr_kernel_features = [200,200,200,200,200,150,100,50]
  hparams.chr_maxpool_size = 3
  return hparams
