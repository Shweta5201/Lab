from __future__ import division
from __future__ import print_function

import os, shutil
import time
import datetime
import os.path
import argparse
import collections

import numpy as np
from sklearn.preprocessing import scale

RATING_MATRIX_FILE = "rating_matrix.txt"
TRAIN_FILE = "train.data"
TEST_FILE= "test.data"

class MovieLensParser():
  
  def __init__(self, data_path, p_data_path,):
    self._data_path = data_path
    self._output_path = p_data_path
    self._num_users = 0
    self._num_items = 0
    self._sequences = None 
    self._rating_matrix = None

  def movielens100k(self):
    # For MovieLens 100K dataset
    # Creating necessary files including training and test data
    data_file = os.path.join(self._data_path,"u.data")
    #for f in os.listdir(self._data_path):
    #  if f.endswith(".base") or f.endswith(".test"):
    #    shutil.copy(os.path.join(self._data_path,f),
    #      self._output_path)
    rating_file = os.path.join(
      self._output_path,RATING_MATRIX_FILE)
    train_file = os.path.join(self._output_path,TRAIN_FILE)
    test_file = os.path.join(self._output_path,TEST_FILE)
    
    # Reading raw data from rating file
    data = list()
    with open(data_file) as rf:
      for line in rf:
        uid, mid, r, ts = line.strip().split("\t")
        data.append((ts, uid, mid, r))
        #data[mid].append((ts, uid, r))
    return np.array(data).astype(int)

  def movielens1M(self):
    # For MovieLens 100K dataset
    # Creating necessary files including training and test data
    data_file = os.path.join(self._data_path,"ratings.dat")

    # Reading raw data from rating file
    data = list()
    with open(data_file) as rf:
      for line in rf:
        uid, mid, r, ts = line.strip().split("::")
        data.append((ts, uid, mid, r))
        #data[mid].append((ts, uid, r))
    return np.array(data).astype(int)

  def movielens10M(self):
    # For MovieLens 100K dataset
    # Creating necessary files including training and test data
    data_file = os.path.join(self._data_path,"ratings.dat")

    # Reading raw data from rating file
    data = list()
    with open(data_file) as rf:
      for line in rf:
        uid, mid, r, ts = line.strip().split("::")
        data.append((ts, uid, mid, r))
    data = np.array(data)
    ratings = sorted(set(data[:,3]))
    ratings = dict(zip(ratings, range(1, len(ratings)+1)))
    data[:,3] = [ratings[r] for r in data[:,3]]
    return data.astype(int)

  def parse_data(self):
    
    #data = self.movielens100k()
    data = self.movielens1M()
    #data = self.movielens10M()
    # Get all the users and items from the data
    users = set(data[:,1])
    movies = set(data[:,2])
    self._num_users = len(users)
    self._num_items = len(movies)
    user2id = items2idx(users)
    movie2id = items2idx(movies)
    
    print(self._num_items)

    train_file = os.path.join(self._output_path,TRAIN_FILE)
    test_file = os.path.join(self._output_path,TEST_FILE)
    rating_file = os.path.join(
      self._output_path,RATING_MATRIX_FILE)
    
    new_data = collections.defaultdict(list)
    for row in data:
      new_data[row[1]].append(
        (row[0],user2id[row[1]],movie2id[row[2]],row[3]))

    train_data = list()
    test_data = list()

    for key, value in new_data.items():
      cutoff = int(round(0.9 * len(value)))
      train_data.extend(value[:cutoff])
      test_data.extend(value[cutoff:])

    train_matrix = np.zeros([len(train_data),4])
    valid_matrix = np.zeros([len(test_data),4])
    with open(train_file,"w") as f:
      for idx,l in enumerate(sorted(train_data)):
        train_matrix[idx] = l
      np.savetxt(f,train_matrix,fmt='%d',delimiter=' | ',
        header="Training data matrix")
    with open(test_file,"w") as f:
      for idx,l in enumerate(sorted(test_data)):
        valid_matrix[idx] = l
      np.savetxt(f,valid_matrix,fmt='%d',delimiter=' | ',
        header="Test data matrix")
    
    # Form user-item rating matrix from the data
    rating_matrix = np.zeros([self._num_users, self._num_items])
    for ts,uid,mid,r in data:
      rating_matrix[user2id[uid]-1,movie2id[mid]-1] = r
    # save numpy user-item rating matrix representation in a file
    with open(rating_file,"w+") as f:
      np.savetxt(f,rating_matrix,fmt='%.1f',delimiter=' | ',
        header="Rating matrix with users as rows and items as columns, indexing starts from 0")
    

def items2idx(items):
  return dict(zip(items, range(1, len(items)+1)))

def main(args):
  print(args)
  # Removing data directory
  if os.path.exists(args.output_dir):
    shutil.rmtree(args.output_dir)
  os.makedirs(args.output_dir)
  if args.dataset == "Movielens":
    parser = MovieLensParser(args.dataset_dir, args.output_dir)
  elif args.dataset == "Netflix":
    parser = NetflixParser(args.dataset_dir, args.output_dir)
  else:
    raise RuntimeError("Unknown dataset!")
  parser.parse_data()

def _parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("dataset", choices=("Movielens","Netflix"),
    default="Movielens")
  parser.add_argument("dataset_dir")
  parser.add_argument("--output-dir", default="../data")
  return parser.parse_args()

if __name__ == '__main__':
  args = _parse_args()
  main(args)

