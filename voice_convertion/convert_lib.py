# -*- coding: utf-8 -*-

import os
import sys
import json
import numpy as np 
import torch as th
import torch.utils.data as data
import torch.optim as optim
import argparse as ap
import operator

from os import path

import dnnVC

import scipy.sparse
import matplotlib.pyplot as plt

import pyworld
import pysptk
#----------------------------------
import argparse

from sklearn.externals import joblib

from sprocket.util import HDF5

# others #############################
def read_feats(listf, ext='mcep'):
    """HDF5 handler
    Create list consisting of arrays listed in the list

    Parameters
    ---------
    listf : str,
        Path of list file
    ext : str,
        `mcep` : mel-cepstrum
        `f0` : F0

    Returns
    ---------
    datalist : list of arrays

    """

    datalist = []
    with open(listf, 'r') as fp:
        for line in fp:
            f = line.rstrip()
            h5f = os.path.join(f + '.h5')
            h5 = HDF5(h5f, mode='r')
            datalist.append(h5.read(ext))
            h5.close()

    return datalist
# training ############################
def train(model, trn_dl, val_dl, tst_dl, device, optimizer, variance, type):
    model.train()
    if (type == 1): # train
        for i, (a, c) in enumerate(trn_dl):
            a, c = a.to(device), c.to(device)
            a.requires_grad_()
            optimizer.zero_grad()
            m_ = model.forward(a)
            loss = -th.mean( dnnVC.log_gaussian_density(0.0, m_, model.variance, c) )
            loss.backward()
            optimizer.step()
    elif (type == 2): # convert
        if th.cuda.is_available():
            trn_dl = trn_dl.cuda()
            m_ = model.forward(trn_dl)
            temp = m_.cpu()
            mean = temp.detach().numpy()
            print(mean.shape)
            y_head = _mlpg(mean, variance)
    
            return y_head

# covert function ########################
def eval(model, trn_dl, val_dl, tst_dl, device, variance):
    model.eval()
    trn_loss = 0.0
    val_loss = 0.0
    tst_loss = 0.0
    with th.no_grad():  
        #i = 0
        for i, (a, c) in enumerate( trn_dl ):
            a, c = a.to(device), c.to(device)
            m_ = model.forward(a)
            loss = -th.sum( dnnVC.log_gaussian_density(0.0,m_, model.variance, c) ).item()
            trn_loss += loss 
            
        #i = 0
        for i,(a,c) in enumerate( val_dl ):
            a,c= a.to(device), c.to(device)
            m_ = model.forward(a)
            loss = -th.sum( dnnVC.log_gaussian_density(0.0,m_, model.variance, c) ).item()
            val_loss += loss 

        for i,(a,c) in enumerate( tst_dl ):
            a,c= a.to(device), c.to(device)
            m_ = model.forward(a)
            loss = -th.sum( dnnVC.log_gaussian_density(0.0,m_, model.variance, c) ).item()
            tst_loss += loss 

    trn_loss /= len(trn_dl.dataset)
    val_loss /= len(val_dl.dataset)
    tst_loss /= len(tst_dl.dataset)
    return trn_loss, val_loss, tst_loss 

def construct_static_and_delta_matrix(T, D, win=[-1.0, 1.0, 0]):
    """Calculate static and delta transformation matrix

    Parameters
    ----------
    T : scala, `T`
        Scala of time length
    D : scala, `D`
        Scala of the number of dimentsion
    win: array, optional, shape (`3`)
        The shape of window matrix for delta.
        Default set to [-1.0, 1.0, 0].

    Returns
    -------
    W : array, shape (`2 * D * T`, `D * T`)
        Array of static and delta transformation matrix.

    """

    static = [0, 1, 0]
    delta = win
    assert len(static) == len(delta)

    # generate full W
    DT = D * T
    ones = np.ones(DT)
    row = np.arange(2 * DT).reshape(2 * T, D)
    static_row = row[::2]
    delta_row = row[1::2]
    col = np.arange(DT)

    data = np.array([ones * static[0], ones * static[1],
                     ones * static[2], ones * delta[0],
                     ones * delta[1], ones * delta[2]]).flatten()
    row = np.array([[static_row] * 3,  [delta_row] * 3]).flatten()
    col = np.array([[col - D, col, col + D] * 2]).flatten()

    # remove component at first and end frame
    valid_idx = np.logical_not(np.logical_or(col < 0, col >= DT))

    W = scipy.sparse.csr_matrix(
        (data[valid_idx], (row[valid_idx], col[valid_idx])), shape=(2 * DT, DT))
    W.eliminate_zeros()

    return W

def _mlpg(mseq, covseq):
    # parameter for sequencial data
    T, sddim = mseq.shape

    # prepare W
    W = construct_static_and_delta_matrix(T, sddim // 2)

    # prepare D
    covseq_ = np.zeros((T, sddim, sddim))
    for i in range(T):
        covseq_[i] = covseq
    D = get_diagonal_precision_matrix(T, sddim, covseq_)

    # calculate W'D
    WD = W.T @ D

    # W'DW
    WDW = WD @ W

    # W'Um
    WDm = WD @ mseq.flatten()

    # estimate y = (W'DW)^-1 * W'Dm
    odata = scipy.sparse.linalg.spsolve(
            WDW, WDm, use_umfpack=False).reshape(T, sddim // 2)

    # return odata
    return odata

def get_diagonal_precision_matrix(T, D, covseq):
    return scipy.sparse.block_diag(covseq, format='csr')
'''
def main(*argv):
    pass

if __name__ == '__main__':
    main()
'''