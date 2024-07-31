import scipy.stats as ss
import numpy as np
from scoring_new import *
from sklearn.metrics import roc_curve, auc
from math import log

def model_eval_new(model,test_data,test_target):

    #Using the model(s) to make predictions
    if type(model) != type([]):
        #If it's not an ensemble
        [pred,loss]=model.predict(test_data)
        scores= np.transpose(pred)[0]
        print(scores.shape)
    else:
        #Otherwise
        scores = scoring_new(model, test_data)
    #Compute the Pearson correlation coefficient
    lable=test_target[0]

    pcc = ss.pearsonr(lable,scores)
    
    # Compute ROC curve and area the curve
    test_label = [1 if aff > 1 - log(500) / log(50000) else 0 for aff in lable]
    fpr, tpr, thresholds = roc_curve(test_label,scores)
    roc_auc = auc(fpr, tpr)
    
    #Compute the accuracy
    threshold = 1 - log(500) / log(50000) 
    predictions = [0 if score < threshold else 1 for score in scores]
    accurate = [1 if predictions[i] == test_label[i] else 0 for i in range(len(predictions))]
    acc = np.sum(accurate)/float(len(accurate))  

    return pcc[0], roc_auc, acc