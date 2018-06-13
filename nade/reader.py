from __future__ import division
from __future__ import print_function

import os
import collections
import numpy as np
from math import ceil
np.set_printoptions(threshold=np.nan)

class Reader():

  def __init__(self, df_path, rm, bs, cs, num_ratings):
    self._rating_matrix = None
    self._num_users = 0
    self._num_items = 0
    self._batch_size = bs
    self._chunk_size = 0
    self._max_seq_len = 0
    self._k = num_ratings

    seq = collections.defaultdict(list)
    with open(df_path) as f:
      for i,line in enumerate(f):
        if i==0:
            continue
        #uid, iid, r, ts = line.strip().split('\t')
        ts, uid, iid, r = line.strip().split(' | ')
        seq[uid].append((ts,uid,iid,r))

    max_len = 0
    _seq = dict()
    for u in seq.keys():
      _seq[int(u)] = (
        np.array(sorted(seq[u])).astype(int),len(seq[u]))
      max_len = max(max_len, _seq[int(u)][1])
    self._max_seq_len = max_len

    if cs == 0:
      self._chunk_size = max_len
    else:
      self._chunk_size = min(cs, max_len)

    self._seq = _seq
    print(len(_seq))
    self._rating_matrix = rm
    self._num_users, self._num_items = self._rating_matrix.shape
    self._users_in_batches = None

  def __getitem__(self, u):
    return self._seq[u]

  @property
  def rating_matrix(self):
    return self._rating_matrix

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
  def sequences(self):
    return self._seq

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def chunk_size(self):
    return self._chunk_size

  @property
  def max_seq_len(self):
    return self._max_seq_len

  def assign_users_to_batches(self, users=None):
    if users is None:
        users = np.random.permutation(self._seq.keys())
    self._users_in_batches = users
    return users

  def u_seq(self, u):
    return self._seq[u][0]

  def u_seq_len(self, u):
    return self._seq[u][1]
    
  def iterate_batches(self):
    """Iterates over batches of user sequences"""
    if self._users_in_batches is None:
        self.assign_users_to_batches()
    users = self._users_in_batches

    def iterate_batch(inputs, targets, seq_lens):
      # Iterates over window chunks
      for j in range(0,targets.shape[1], self.chunk_size):
        c_sizes = np.minimum(self.chunk_size, seq_lens)
        seq_lens = np.maximum(0, seq_lens - c_sizes)
        yield (inputs[:,j:j+self.chunk_size],
          targets[:,j:j+self.chunk_size], c_sizes)
      
    for i in range(0, len(users), self.batch_size):
      batch = users[i:i+self.batch_size]
      _len = max([self.u_seq_len(u) for u in batch])
      q,r = divmod(_len, self.chunk_size)
      _len = (q + ((r>0)&1)) * self.chunk_size
      inputs = np.zeros([len(batch), _len, 4])
      targets = np.zeros([len(batch), _len])
      seq_lengths = np.zeros([len(batch)])
      # Forming batch data: inputs, labels, sequence lengths
      for idx,u in enumerate(batch):
        # rmu = self.rating_matrix[u - 1]
        s,l = self[u]
        s = np.random.permutation(s)
        inputs[idx,1:l,:3] = s[:l-1,1:]
        inputs[idx,:l,3] = s[:,2]
        targets[idx,:l] = s[:,3]
        seq_lengths[idx] = l
      yield batch,iterate_batch(inputs,targets,seq_lengths)

  @classmethod
  def get_rating_matrix(cls, rmf_path):
    rating_matrix = None
    with open(rmf_path) as rm:
      rating_matrix = np.loadtxt(rm, 'float', delimiter=' | ').astype(int)
    return rating_matrix
