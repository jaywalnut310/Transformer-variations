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

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

import tensorflow as tf
from .generator_utils import get_or_generate_vocab

_KOEN_TRAIN_DATASETS = [("dict.ko.train", "dict.en.train")]
_KOEN_TEST_DATASETS = [("dict.ko.valid", "dict.en.valid")]

def get_or_compile_data(tmp_dir, datasets, filename):
  """Concatenate all `datasets` and save to `filename`."""
  filename = os.path.join(tmp_dir, filename)

  if (tmp_dir is not None) \
        and (tf.gfile.Exists(filename + ".lang1")) \
        and (tf.gfile.Exists(filename + ".lang2")):
    return filename

  with tf.gfile.GFile(filename + ".lang1", mode="w") as lang1_resfile:
    with tf.gfile.GFile(filename + ".lang2", mode="w") as lang2_resfile:
      for dataset in datasets:
        lang1_filename, lang2_filename = dataset
        lang1_filepath = os.path.join(tmp_dir, lang1_filename)
        lang2_filepath = os.path.join(tmp_dir, lang2_filename)

        with tf.gfile.GFile(lang1_filepath, mode="r") as lang1_file:
          with tf.gfile.GFile(lang2_filepath, mode="r") as lang2_file:
            lines1, lines2 = lang1_file.readlines(), lang2_file.readlines()

            for line1 in lines1:
              lang1_resfile.write(line1)
            for line2 in lines2:
              lang2_resfile.write(line2)

  return filename

@registry.register_problem
class TranslateKoenChar2wordSimple(translate.TranslateProblem):
  """Problem spec for Simple Ko-En translation."""

  @property
  def sourced_vocab_size(self):
    return None 

  @property
  def targeted_vocab_size(self):
    return 2**13 #8192

  @property
  def source_vocab_name(self):
    return "vocab.ko"

  @property
  def target_vocab_name(self):
    return "vocab.en"      

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
    datasets = _KOEN_TRAIN_DATASETS if train else _KOEN_TEST_DATASETS
    source_datasets = [item[0] for item in _KOEN_TRAIN_DATASETS]
    target_datasets = [item[1] for item in _KOEN_TRAIN_DATASETS]

    tf.gfile.MakeDirs(data_dir)

    source_vocab = get_or_generate_vocab(data_dir, tmp_dir, self.source_vocab_name, 
            self.sourced_vocab_size, source_datasets, mode='character')
    target_vocab = get_or_generate_vocab(data_dir, tmp_dir, self.target_vocab_name,
            self.targeted_vocab_size, target_datasets)

    tag = "train" if train else "dev"
    data_path = get_or_compile_data(tmp_dir, datasets, "simple_koen_tok_%s" % tag)

    return translate.bi_vocabs_token_generator(data_path + ".lang1",
            data_path + ".lang2", source_vocab, target_vocab, text_encoder.EOS_ID)

  def feature_encoders(self, data_dir):
    source_vocab_filename = os.path.join(data_dir, self.source_vocab_name)
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.TokenTextEncoder(source_vocab_filename, replace_oov="UNK")
    target_token = text_encoder.TokenTextEncoder(target_vocab_filename, replace_oov="UNK")
    return {
      "inputs": source_token,
      "targets": target_token,
    }




