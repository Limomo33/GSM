from __future__ import print_function

import scipy.sparse as sp
import numpy as np
from numpy import random,mat
from scipy.sparse.linalg.eigen.arpack import eigsh, ArpackNoConvergence
import scipy.stats as ss
from scoring import *
from math import log

def parse_label(file):
    seq = []
    f=open(file, 'r').readlines()
    for l in f:
        if '>'is not l[0]:
            seq.append(float(l.strip('\n')))
    return seq

def read(path):
    f = open(path,"r")
    blosum = []
    for line in f:
        #blosum.append(float(i) for i in line.split())
        blosum.append([(float(i)) for i in line.split()])
    f.close()
    #print(blosum)
    return np.array(blosum)

def load_adj(path="/data1/lrm1/MHC/ACME_codes/",adj="adj.txt"):

    """Load citation network dataset (cora only for now)"""
    print('Loading {} adj...')
    a=read(path+adj)
    edges=list()
    for i, s in enumerate(a):
        for j, bp in enumerate(s):
            if a[i][j]==1:
                edges.append([i,j])
    edges=np.array(edges)
    adj = sp.csr_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(a.shape[1], a.shape[1]), dtype=np.float32)

    return adj

def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0)  ##D=matrix D_-0.5_hat
        a_norm = adj.dot(d).transpose().dot(d).tocsr()   ##D*A*D
    else:
        #d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = adj.tocsr()
    return a_norm


def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize_adj(adj, symmetric)
    return adj

def creat_adj():
    A=np.zeros((300,300))
    A_s=np.zeros((24,180))
    _=np.zeros((24,120))
    A_p=np.zeros((24,24))
    step=18
    len=9
    #总长为180，以18为区域进行切分
    # 从前往后建立临接关系
    for i in range(12):
        for j in range(10):
            A_s[i][i+j*step:i+j*step+len]=1
            A_s[i+12][i+j*step:i+j*step+len]=1
    A_s=np.hstack((A_s, _))
    A_st=A_s.T
    aa = np.vstack((np.hstack((A, A_st)), np.hstack((A_s, A_p))))
    print(np.shape(aa))
    return (aa)
