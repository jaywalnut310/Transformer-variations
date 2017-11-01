from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.data_generators import text_encoder

class CharacterTextEncoder(text_encoder.TokenTextEncoder):
  """Encoder based on a user-supplied vocabulary (file or list)."""

  def __init__(self,
               vocab_filename,
               reverse=False,
               vocab_list=None,
               replace_oov=None,
               num_reserved_ids=NUM_RESERVED_TOKENS):
    """Initialize from a file or list, one token per line.

    Handling of reserved tokens works as follows:
    - When initializing from a list, we add reserved tokens to the vocab.
    - When initializing from a file, we do not add reserved tokens to the vocab.
    - When saving vocab files, we save reserved tokens to the file.

    Args:
      vocab_filename: If not None, the full filename to read vocab from. If this
         is not None, then vocab_list should be None.
      reverse: Boolean indicating if tokens should be reversed during encoding
         and decoding.
      vocab_list: If not None, a list of elements of the vocabulary. If this is
         not None, then vocab_filename should be None.
      replace_oov: If not None, every out-of-vocabulary token seen when
         encoding will be replaced by this string (which must be in vocab).
      num_reserved_ids: Number of IDs to save for reserved tokens like <EOS>.
    """
    super(CharacterTextEncoder, self).__init__(vocab_filename, reverse, vocab_list, replace_oov, num_reserved_ids)


  def encode(self, sentence):
    """Converts a non space-separated string of tokens to a list of ids."""
    tokens = list(sentence.strip())
    if self._replace_oov is not None:
      tokens = [t if t in self._token_to_id else self._replace_oov
                for t in tokens]
    ret = [self._token_to_id[tok] for tok in tokens]
    return ret[::-1] if self._reverse else ret

  def decode(self, ids):
    return "".join(self.decode_list(ids))
