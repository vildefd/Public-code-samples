# -*- coding: utf-8 -*-
 #%%

import numpy as np 
from vdr_accuracy import acc#, confmatr
from sklearn.metrics import confusion_matrix
from dtw import dtw # dynamic time warp library
import collections # to count most common

def score_metrics(confusion_matrix):    
    # Confusion matrix cheat sheet:
    # tn = (0, 0)
    # fn = (1, 0)
    # tp = (1, 1)
    # fp = (0, 1)
    precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1]) 
    tpr = confusion_matrix[ 1, 1] / (confusion_matrix[ 1, 1] + confusion_matrix[ 1, 0]) 
    tnr = confusion_matrix[ 0, 0] / (confusion_matrix[ 0, 0] + confusion_matrix[ 0, 1])
    balanced_acc = (tpr + tnr) / 2
    f1_score = 2 * (precision * tpr) / (precision + tpr)

    if np.isnan(balanced_acc):
        balanced_acc = 0
    if np.isnan(f1_score):
        f1_score = 0

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)
    return balanced_acc, f1_score, accuracy

def validate_knn(distances, y_val, val_indices_1, val_indices_2, k):

    M_val = len(val_indices_2)
    #M_te = len(test_indices_2)

    N_val = len(y_val)
    #N_te = len(y_te)

    #Construct train+validation distance grid
    val_set = np.zeros((N_val, M_val))
    for i in range(N_val):
        for j in range(M_val):
            val_set[i, j] = distances[ val_indices_1[i], val_indices_2[j] ]
    
    val_set = val_set/np.max(distances)

    nearestneighborindexes = []

    predictions_val = np.zeros_like(y_val)

    for n in range(N_val):

        # Find the k nearest neighbors, excluding itself (which has distance 0 (zero))
        nearestneighborindexes = np.argsort(val_set[n, :])[1:k+1]
        
        # Find the most common label from the set of k nearest neighbors, and assign it
        label_count =  collections.Counter( y_val[nearestneighborindexes] )
        
        most_common_label = int(label_count.most_common(1)[0][0])
        
        predictions_val[n] = most_common_label
        label_count.clear()

    confusion = confusion_matrix(y_val, predictions_val)
    balanced_acc, f1_score, accuracy = score_metrics(confusion)
    
    return  accuracy, balanced_acc, f1_score

def test_knn(distances, y_val, y_te, test_indices_1, val_indices_2, best_k):
    
    nearestneighborindexes = []

    N_te = len(test_indices_1)
    M_te = len(val_indices_2)

    test_set = np.zeros((N_te, M_te))
    
    for i in range(N_te):
        for j in range(M_te):
            test_set[i, j] = distances[test_indices_1[i], val_indices_2[j] ]
    
    test_set = test_set / np.max(distances)

    best_k = int(best_k)
    predictions_te = np.zeros_like(y_te)
    for n in range(N_te):
        
        # Find the k nearest neighbors, excluding itself (which has distance 0 (zero))
        nearestneighborindexes = np.argsort(test_set[n,:])[1:best_k+1]

        # Find the most common label from the set of k nearest neighbors, and assign it
        label_count =  collections.Counter( y_val[nearestneighborindexes] )
        
        most_common_label = int(label_count.most_common(1)[0][0])
        
        predictions_te[n] = most_common_label
        label_count.clear()

    confusion_data = confusion_matrix(y_te, predictions_te)
    balanced_acc, f1_score, acc = score_metrics(confusion_data)

    return f1_score, balanced_acc, acc


    