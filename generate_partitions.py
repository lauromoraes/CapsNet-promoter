#!/usr/bin/env python
"""
Keras implementation of CapsNet in Hinton's paper Dynamic Routing Between Capsules.

Usage:
       python CapsNet.py
       python CapsNet.py --epochs 100
       python CapsNet.py --epochs 100 --num_routing 3
       ... ...

    
"""
import numpy as np
import pandas as pd
np.random.seed(1337)

from keras import layers, models, optimizers
from keras import backend as K
from capsulelayers import CapsuleLayer, PrimaryCap, Length, Mask
from keras.preprocessing import sequence
import keras.callbacks as cbks
from sklearn.model_selection import StratifiedShuffleSplit
import os

def load_dataset(organism):
    from ml_data import SequenceNucsData
    
    print('Load organism: {}'.format(organism))
    npath, ppath = './fasta/{}_neg.fa'.format(organism), './fasta/{}_pos.fa'.format(organism)
    print(npath, ppath)
    
    samples = SequenceNucsData(npath, ppath, k=k)
    
    X, y = samples.getX(), samples.getY()
#    X = X.reshape(-1, 38, 79, 1).astype('float32')
    X = X.astype('int32')
    y = y.astype('int32')
    print('Input Shapes\nX: {} | y: {}'.format(X.shape, y.shape))
    maxlen = X.shape[1]
    return X, y

def load_partition(train_index, test_index, X, y):
    x_train = X[train_index,:]
    y_train = y[train_index]
    
    x_test = X[test_index,:]
    y_test = y[test_index]
    
#    y_train = to_categorical(y_train.astype('float32'))
#    y_test = to_categorical(y_test.astype('float32'))
    
    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":
    import os    
    import numpy as np    
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--organism', default=None, help="The organism used for test. Generate auto path for fasta files. Should be specified when testing")
    parser.add_argument('-p', '--partitions', default=5, type=int, help="Number of pair (train, test) partitions to be created.")
    parser.add_argument('-k', '--kmer', default=1, type=int, help="Length of kmer created.")
    args = parser.parse_args()

    # Create output directory
    save_dir = '{}_{}-mer_{}-partition'.format(args.organism, args.partitions)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Load data
    X, y = load_dataset(args.organism)
    # Generate partitions    
    kf = StratifiedShuffleSplit(n_splits=args.partitions, random_state=34267)
    kf.get_n_splits(X, y)
    
    runStep = 0
     
    for train_index, test_index in kf.split(X, y):
        runStep+=1
        (x_train, y_train), (x_test, y_test) = load_partition(train_index, test_index, X, y)
        print(x_train.shape)
        print(y_train.shape)
    
    print('END')
