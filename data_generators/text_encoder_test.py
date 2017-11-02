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

"""Tests for tensor2tensor.data_generators.text_encoder."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import collections
import io
import os
import shutil

# Dependency imports
import mock
import six

from tensor2tensor.data_generators import text_encoder
import tensorflow as tf
from text_encoder import CharacterTextEncoder

class CharacterTextEncoderTest(tf.test.TestCase):

  @classmethod
  def setUpClass(cls):
    """Make sure the test dir exists and is empty."""
    cls.test_temp_dir = os.path.join(tf.test.get_temp_dir(), "encoder_test")
    shutil.rmtree(cls.test_temp_dir, ignore_errors=True)
    os.mkdir(cls.test_temp_dir)

  def test_character_encdec_from_non_space_separated_sentence(self):
    corpus = "ABCDEFGHI "
    vocab_filename = os.path.join(self.test_temp_dir, "abc.vocab")

    # Make text encoder from a list and store vocab to fake filesystem.
    encoder = CharacterTextEncoder(None, vocab_list=list(corpus))

    sentence = "ABC DEF GH I"
    s_enc = encoder.encode(sentence)
    s_dec = encoder.decode(s_enc)

    self.assertEqual(len(list(sentence)), len(s_enc))
    self.assertEqual(sentence, s_dec)

if __name__ == "__main__":
  tf.test.main()
