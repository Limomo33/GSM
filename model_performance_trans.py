from foutput import *
from model_eval import *
import numpy as np
from seq_distance import *
from Trans_matrix import *
def model_performance(models, test_dict, d, global_args, alleles = None):

    [blosum_matrix, aa, main_dir, output_path] = global_args
    performance_dict = {}
    for allele in sorted(test_dict.keys()):
        #See if we are testing on specified alleles
        if alleles is not None and allele not in alleles:
            continue
        #We only test on datasets with >= 10 data
        if len(test_dict[allele]) < 10:
            continue
        foutput(allele+" "+str(len(test_dict[allele])), output_path)
        print (allele)
        # A1 = np.zeros((300, 300))
        # A2 = np.ones((300, 24))
        # A3 = np.ones((24, 300))
        # A4 = np.zeros((24, 24))
        # a = np.vstack((np.hstack((A1, A2)), np.hstack((A3, A4))))
        # dd = np.array(a.sum(1))

        [test_pep, test_mhc, test_target] = [[i[j] for i in test_dict[allele]] for j in range(3)]
        #test_merge = np.concatenate((np.array(test_mhc), np.array(test_pep)[:, :12, :]), axis=1)
        #Evaluate the performance of our model.
        pcc, roc_auc, max_acc = model_eval(models,[np.array(test_pep),np.array(test_mhc),np.tile(d,(len(test_pep),1,1))],#,np.tile(a,(len(test_pep),1,1)),np.tile(d,(len(test_pep),1,1))],
                                            [np.array(test_target)])
        performance_dict[allele] = [pcc, roc_auc, max_acc]
        
    return performance_dict