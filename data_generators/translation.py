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

# Dependency imports

from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import translate_enzh 
from tensor2tensor.data_generators import translate_ende
from tensor2tensor.utils import registry

import tensorflow as tf

EOS = text_encoder.EOS_ID

# Renew SpaceIDs
problem.SpaceID.ZH_CHR = 31

# Datasets
_GENERIC_TRAIN_DATASETS = [
    [
        "http://dumb.dumb/train.tar",
        ("train.lang1", "train.lang2"),
    ],
]

_GENERIC_TEST_DATASETS = [
    [
        "http://dumb.dumb/test.tar",
        ("dev.lang1", "dev.lang2"),
    ],
]


def char2word_generator(source_path, target_path, source_character_vocab, target_token_vocab, eos=None):
  """Generator for sequence-to-sequence tasks that uses tokens.

  This generator assumes the files at source_path and target_path have
  the same number of lines and yields dictionaries of "inputs" and "targets"
  where inputs are token ids from the " "-split source (and target, resp.) lines
  converted to integers using the token_map.

  Args:
    source_path: path to the file with source sentences.
    target_path: path to the file with target sentences.
    source__vocab: text_encoder.TextEncoder object.
    target_token_vocab: text_encoder.TextEncoder object.
    eos: integer to append at the end of each sequence (default: None).

  Yields:
    A dictionary {"inputs": source-line, "targets": target-line} where
    the lines are integer lists converted from tokens in the file lines.
  """

  eos_list = [] if eos is None else [eos]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      while source and target:
        source_ints = source_character_vocab.encode(source.strip()) + eos_list
        target_ints = target_token_vocab.encode(target.strip()) + eos_list
        yield {"inputs": source_ints, "targets": target_ints}
        source, target = source_file.readline(), target_file.readline()


@registry.register_problem
class TranslateEndeWmtChar2word8k(translate.TranslateProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13 # 8192

  @property
  def target_vocab_name(self):
    return "vocab.ende-de.%d" % self.targeted_vocab_size

  def generator(self, data_dir, tmp_dir, train):
    source_character_vocab = text_encoder.ByteTextEncoder()
    datasets = translate_ende._ENDE_TRAIN_DATASETS if train else translate_ende._ENDE_TEST_DATASETS
    target_datasets = [[item[0], [item[1][1]]] for item in translate_ende._ENDE_TRAIN_DATASETS]
    
    target_token_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.target_vocab_name, self.targeted_vocab_size,
        target_datasets)
    
    tag = "train" if train else "dev"
    data_path = translate.compile_data(tmp_dir, datasets, "translate_ende_char2word_%s" % tag)
    return char2word_generator(data_path + ".lang1", data_path + ".lang2",
            source_character_vocab, target_token_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.EN_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.DE_TOK

  def feature_encoders(self, data_dir):
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.ByteTextEncoder()
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }


@registry.register_problem
class TranslateEnzhWmtCharacters(translate.TranslateProblem):
    """Problem spec for WMT En-Zh translation."""

    @property
    def is_character_level(self):
        return True

    @property
    def num_shards(self):
        return 10 # This is a small dataset.

    def generator(self, data_dir, tmp_dir, train):
        character_vocab = text_encoder.ByteTextEncoder()
        datasets = _ZHEN_TRAIN_DATASETS if train else _ZHEN_TEST_DATASETS
        tag = "train" if train else "dev"
        data_path = translate.compile_data(tmp_dir, datasets, "translate_enzh_chr_%s" % tag)
        return character_generator(data_path + ".lang1", data_path + ".lang2",
                                  character_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.EN_TOK

    @property
    def target_space_id(self):
        return problem.SpaceID.ZH_TOK

@registry.register_problem
class TranslateGenericCharacters(translate.TranslateProblem):
    """Problem spec for GENERIC Character-level Translation."""

    @property
    def is_character_level(self):
        return True

    def generator(self, data_dir, tmp_dir, train):
        character_vocab = text_encoder.ByteTextEncoder()
        datasets = _GENERIC_TRAIN_DATASETS if train else _GENERIC_TEST_DATASETS
        tag = "train" if train else "dev"
        data_path = translate.compile_data(data_dir, datasets, "generic_chr_%s" % tag)
        return translate.character_generator(data_path + ".lang1", data_path + ".lang2", character_vocab, EOS)

    @property
    def input_space_id(self):
        return problem.SpaceID.GENERIC

    @property
    def target_space_id(self):
        return problem.SpaceID.GENERIC

@registry.register_problem
class TranslateZhenWmtChar2word8k(translate.TranslateProblem):
  """Problem spec for WMT En-De translation."""

  @property
  def targeted_vocab_size(self):
    return 2**13 # 8192

  @property
  def target_vocab_name(self):
    return "vocab.zhen-en.%d" % self.targeted_vocab_size

  def generator(self, data_dir, tmp_dir, train):
    source_character_vocab = text_encoder.ByteTextEncoder()
    datasets = translate_enzh._ZHEN_TRAIN_DATASETS if train else translate_enzh._ZHEN_TEST_DATASETS
    target_datasets = [[item[0], [item[1][1]]] for item in translate_enzh._ZHEN_TRAIN_DATASETS]
    
    target_token_vocab = generator_utils.get_or_generate_vocab(
        data_dir, tmp_dir, self.target_vocab_name, self.targeted_vocab_size,
        target_datasets)
    
    tag = "train" if train else "dev"
    data_path = translate.compile_data(tmp_dir, datasets, "translate_zhen_char2word_%s" % tag)
    return char2word_generator(data_path + ".lang1", data_path + ".lang2",
            source_character_vocab, target_token_vocab, EOS)

  @property
  def input_space_id(self):
    return problem.SpaceID.ZH_CHR

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_TOK

  def feature_encoders(self, data_dir):
    target_vocab_filename = os.path.join(data_dir, self.target_vocab_name)
    source_token = text_encoder.ByteTextEncoder()
    target_token = text_encoder.SubwordTextEncoder(target_vocab_filename)
    return {
        "inputs": source_token,
        "targets": target_token,
    }


