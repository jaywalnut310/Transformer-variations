from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import Counter
import os

from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import tokenizer
from tensor2tensor.data_generators import generator_utils

import tensorflow as tf
from .text_encoder import CharacterTextEncoder

def get_or_generate_vocab(data_dir, tmp_dir, vocab_filename, vocab_size, text_files, mode='word'):
  """Implementation for vocab generators.
  Args:
    data_dir: The base directory where data and vocab files are stored. If None,
        then do not save the vocab even if it doesn't exist.
    ...
    vocab_filename: relative filename where vocab file is stored
    vocab_size: None is accepted. target size of the vocabulary constructed by TokenTextEncoder
    ...
  Returns:
    A TokenTextEncoder vocabulary object.
  """

  def generate():
    tf.logging.info("Generating vocab from: %s", str(text_files))
    for lang_file in text_files:
      tf.logging.info("Reading file: %s" % lang_file)
      filepath = os.path.join(tmp_dir, lang_file)

      with tf.gfile.GFile(filepath, mode="r") as source_file:
        for line in source_file:
          line = line.strip()
          yield line

  def encode(text):
    if mode=='character':
      return list(text)
    else:
      return tokenizer.encode(text)

  def text_encoder(vocab_filepath, replace_oov):
    if mode=='character':
      return CharacterTextEncoder(vocab_filepath, replace_oov=replace_oov)
    else:
      return text_encoder.TokenTextEncoder(vocab_filepath, replace_oov="UNK")

  if data_dir is None:
    vocab_filepath = None
  else:
    vocab_filepath = os.path.join(data_dir, vocab_filename)

  if vocab_filepath is not None and tf.gfile.Exists(vocab_filepath):
    tf.logging.info("Found vocab file: %s", vocab_filepath)
    vocab = text_encoder(vocab_filepath, replace_oov="UNK")
    return vocab

  tf.logging.info("Generating vocab file: %s", vocab_filepath)
  token_counts = Counter()
  for item in generate():
    for tok in encode(text_encoder.native_to_unicode(item)):
      token_counts[tok] += 1

  with tf.gfile.GFile(vocab_filepath, mode="w") as f:
    word_list = list(map(lambda x: x[0], token_counts.most_common()))
    word_list = ['UNK'] + word_list
    if vocab_size is not None:
      word_list = word_list[:vocab_size]
    for word in word_list:
      f.write(word + '\n')

  vocab = text_encoder(vocab_filepath, replace_oov="UNK")

  return vocab

