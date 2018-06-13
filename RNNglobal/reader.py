from __future__ import division
from __future__ import print_function

import os
import collections
import numpy as np
from math import ceil
np.set_printoptions(threshold=np.nan)

class Reader():

  def __init__(self, df_path, bs, cs, num_ratings,
    num_users, num_items):

    self._data_matrix = None
    self._num_users = 0
    self._num_items = 0
    self._batch_size = bs
    self._chunk_size = cs 
    self._k = num_ratings

    with open(df_path) as f:
      self._data_matrix = np.loadtxt(f, 'float', delimiter=' | ').astype(int)

    self._max_seq_len = self._data_matrix.shape[0]
    self._num_users = num_users
    self._num_items = num_items

  def __getitem__(self, u):
    return self._seq[u]

  @property
  def data_matrix(self):
    return self._data_matrix

  @property
  def num_users(self):
    return self._num_users

  @property
  def num_items(self):
    return self._num_items
  
  @property
  def k(self):
    return self._k

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def chunk_size(self):
    return self._chunk_size

  @property
  def max_seq_len(self):
    return self._max_seq_len
 
  def iterate_batches(self):
    """Iterates over batches of user sequences"""

    def iterate_batch(inputs, targets, seq_lens):
      # Iterates over window chunks
      print(seq_lens)
      for j in range(0,targets.shape[1], self.chunk_size):
        c_sizes = np.minimum(self.chunk_size, seq_lens)
        seq_lens = np.maximum(0, seq_lens - c_sizes)
        yield (inputs[:,j:j+self.chunk_size],
          targets[:,j:j+self.chunk_size], c_sizes)
      
    q,r = divmod(self.max_seq_len,self.chunk_size)
    _len = (q + ((r>0)&1)) * self.chunk_size
    # Forming batch data: inputs, labels, sequence lengths
    inputs = np.zeros([self.batch_size, _len, 2])
    targets = np.zeros([self.batch_size, _len])
    inputs[:,:self.max_seq_len,:] = self._data_matrix[:,1:3].reshape(
      [1,self.max_seq_len,2])
    targets[:,:self.max_seq_len] = self._data_matrix[:,3].reshape(
      [1,self.max_seq_len])
    seq_lengths = np.array([self.max_seq_len])
    yield iterate_batch(inputs,targets,seq_lengths)

  @classmethod
  def get_rating_matrix(cls, rmf_path):
    rating_matrix = None
    with open(rmf_path) as rm:
      rating_matrix = np.loadtxt(rm, 'float', delimiter=' | ').astype(int)
    return rating_matrix
