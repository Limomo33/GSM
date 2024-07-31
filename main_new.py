import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import numpy as np
import random
from sklearn.metrics import roc_curve, auc
import re
import copy
import scipy.stats as ss
from math import *
import keras
import pickle
from keras.models import Sequential
from keras.layers import *
from keras.models import Model
import tensorflow as tf
from keras import backend as K
from sklearn.model_selection import KFold
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.visible_device_list = "3"
sess = tf.Session(config=config)
K.set_session(sess)

from main_model_training import main_model_training
from read_blosum import read_blosum
from main_cross_validation import *
from main_external_testing import *
from main_test_attention import *
from main_motif import *
from main_leave_one_out import *
from main_MHC_clustering import *
#from main_test_other_data import *
from main_pearson_benchmark_redundancy import *
from main_anti_anchor import *
#from main_cross_validation import *
#from main_cross_validation_attention_fc import *
# from main_cross_validation_gru import *
# from main_cross_validation_without_CNN import *
# from main_cross_validation_xz import *
# from main_cross_validation_OneHot import *
#from main_cross_validation_transformer import *
from main_cross_validation_new_test import *
# from main_cross_validation_gat import *
from main_test_case_data import *
#from l2_train import *
from main_binding_prediction import *
#from view import *
#Output
from foutput import *
from read_pp import *

def main(func_ind):
    #Path to the Attention_CNN folder
    #main_dir = "E:/MHC/ACME/ACME/ACME-master/ACME_codes/"
    main_dir="/data1/lrm1/MHC/ACME_codes/"
    aa={"A":0,"R":1,"N":2,"D":3,"C":4,"Q":5,"E":6,"G":7,"H":8,"I":9,"L":10,"K":11,"M":12,"F":13,"P":14,"S":15,"T":16,"W":17,"Y":18,"V":19}     
    #Load the blosum matrix for encoding
    path_blosum = main_dir + r"blosum50.txt"
    blosum_matrix = read_blosum(path_blosum)

    output_path=main_dir+"results/new.txt"
    main_cross_validation_new([blosum_matrix, aa, main_dir, output_path])

        
