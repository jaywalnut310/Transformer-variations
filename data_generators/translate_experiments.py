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

"""Data generators for translation data-sets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf


_TRAIN_DATASET = ("src.trn", "tgt.trn")
_DEV_DATASET = ("src.dev", "tgt.dev")

@registry.register_problem
class TranslateExp(translate.TranslateProblem):
  """Problem spec for Experiments."""

  @property
  def shared_vocab_name(self):
    return "vocab.shared"

  @property
  def input_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def target_space_id(self):
    return problem.SpaceID.GENERIC

  @property
  def num_shards(self):
    return 10  # This is a small dataset.

  def generator(self, data_dir, tmp_dir, train):
    tf.gfile.MakeDirs(data_dir)

    vocab_filepath_old = os.path.join(tmp_dir, self.shared_vocab_name)
    vocab_filepath_new = os.path.join(data_dir, self.shared_vocab_name)
    tf.gfile.Copy(vocab_filepath_old, vocab_filepath_new, overwrite=True)
    shared_vocab = text_encoder.TokenTextEncoder(vocab_filepath_new, replace_oov='<unk>')

    tag = "trn" if train else "dev"

    source_path = os.path.join(tmp_dir, "src.%s" % tag)
    target_path = os.path.join(tmp_dir, "tgt.%s" % tag)
    return translate.token_generator(source_path, target_path, shared_vocab, text_encoder.EOS_ID)

  def feature_encoders(self, data_dir):
    shared_vocab_filepath = os.path.join(data_dir, self.shared_vocab_name)
    shared_token = text_encoder.TokenTextEncoder(shared_vocab_filepath, replace_oov='<unk>')
    return {
      "inputs": shared_token,
      "targets": shared_token,
    }

