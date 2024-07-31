import re
import numpy as np
from math import log

def read_validation_data(path,pseq_dict, global_args):
    [blosum_matrix, aa, main_dir, output_path] = global_args
    data = []
    target = []
    #c=[]
    f = open(path,"r")
    for line in f:
        info = re.split("\t",line)#Retrive information from a tab-delimited line
        allele = info[1]
        if allele in pseq_dict.keys():
            affinity = 1-log(float(info[5]))/log(50000)
            pep = info[3]#Sequence of the peptide in the form of a string, like "AAVFPPLEP"
            pep_blosum = []#Encoded peptide seuqence
            for residue_index in range(12):
                #Encode the peptide sequence in the 1-12 columns, with the N-terminal aligned to the left end
                #If the peptide is shorter than 12 residues, the remaining positions on
                #the rightare filled will zero-padding
                if residue_index < len(pep):
                    #c=blosum_matrix[aa[pep[residue_index]]]+[i for i in pp_matrix[aa[seq_dict[allele][index]]]]
                    pep_blosum.append(blosum_matrix[aa[pep[residue_index]]])

                else:
                    pep_blosum.append(np.zeros(20))
            for residue_index in range(12):
                #Encode the peptide sequence in the 13-24 columns, with the C-terminal aligned to the right end
                #If the peptide is shorter than 12 residues, the remaining positions on
                #the left are filled will zero-padding
                if 12 - residue_index > len(pep):
                    pep_blosum.append(np.zeros(20)) 
                else:
                    pep_blosum.append(blosum_matrix[aa[pep[len(pep) - 12 + residue_index]]])
            new_data = [pep_blosum, pseq_dict[allele]]
            data.append(new_data)
            target.append(affinity)
    return data, target