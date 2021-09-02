# -*- coding: utf-8 -*-
 #%%


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import random
import os, os.path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from tensorflow.keras import callbacks
#import timeit
from common_parameters import imputation_method_list
import tensorflow.keras.utils
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LayerNormalization
from tensorflow.keras import Input, Model
from tensorflow.keras.callbacks import EarlyStopping
import tcn
from tcn import TCN, tcn_full_summary
from vdr_imputation import Imputation
import time

def score_metrics(confusion_matrix):
    precision = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1]) 
    tpr = confusion_matrix[ 1, 1] / (confusion_matrix[ 1, 1] + confusion_matrix[ 1, 0]) 
    tnr = confusion_matrix[ 0, 0] / (confusion_matrix[ 0, 0] + confusion_matrix[ 0, 1])
    balanced_acc = (tpr + tnr) / 2
    f1_score = 2 * (precision * tpr) / (precision + tpr)
    accuracy = (confusion_matrix[0,0] + confusion_matrix[1, 1])/ np.sum(confusion_matrix)
    return balanced_acc, f1_score, accuracy


#Classification - TCN


Y = {}
y = []

synth = True# whether or not to use fake data

if not synth:
    Y = loadmat('../data_src/SSI_data/Y.mat')
    y = np.ravel(Y['Y'])

else:
    Y = pd.read_csv('../data_src/synth_data/synth_y.csv', delimiter='\t', header=None)
    y = np.ravel(Y.values)
    for i in range(len(y)):
        if y[i] == 1:
            y[i] = 0
        if y[i] == 2:
            y[i] = 1
#string paths
figure_path = '../figures/plots/'
if not os.path.exists(figure_path):
    os.makedirs(figure_path)

folder = r'results\\TCN\\'
if not os.path.exists(folder):
    os.makedirs(folder)


#Create informative missingness 
N = len(y)#total number of patients
test_names = ["Hemoglobin", "Leukocytes", "CRP", "Potassium", "Sodium", "Creatinine", "Thromocytes", "Albumin", "Carbamide", "Glucose", "Amylase"]
num_dims = 0
num_steps = 0



time_index = []


validation_runs = 20 #10
indices = [i for i in range(len(y))]
splitter = StratifiedShuffleSplit(n_splits=2, train_size=0.8)
train_indices_array = np.zeros((validation_runs, int(np.round(N*0.8))))
test_indices_array = np.zeros((validation_runs, int(np.round(N*0.2))))


for i in range(validation_runs):
    train_indices_array[i], test_indices_array[i] = next( iter( splitter.split(indices, y)))
train_indices_array = np.array(train_indices_array, dtype=int)
test_indices_array = np.array(test_indices_array, dtype=int)

for method in imputation_method_list:
    
    x = []
    if not synth:
        X = loadmat('../SSI_data/X.mat')
        x = X['X']
        num_dims = len(test_names)
        

    else:
        x = pd.read_csv('synth_x.csv', delimiter='\t', header=None, engine='python').values
        num_dims = 3
    
    x = np.reshape(x, (len(y), num_dims, -1))
    num_steps = np.shape(x)[2]
    time_index = np.array([i for i in range(num_steps)], dtype=int)

    #Imputation, if wanted
    x_imp = Imputation(x)
    x_imputed = []


    if method == 'zero':
        x_imputed = x 
    else:
        x_imp.impute(method)
        x_imputed = x_imp.data()
    
    

    #Init
    y_tr = []
    y_te = []
    y_val = []

    N_tr = 0
    M_tr = 0

    N_te = 0
    M_te = 0

    N_val = 0
    M_val = 0

    count_between_missing_matrix = np.zeros_like(x)
    indicate_missingness_matrix = np.zeros_like(x)

    indicate_missingness_matrix = np.array(x[:,:,:] == 0, dtype=int)
    x = x_imputed

    for n in range(N):
        for i in range(num_dims):
            count = 0
            for j in time_index:

                if indicate_missingness_matrix[n, i, j]:
                    count += 1
                else:
                    count = 0
                
                count_between_missing_matrix[n, i, j] = count   

    new_x = np.zeros((N, num_steps, 3*num_dims))

    for i in range(N):
        new_x[i, :, 0:num_dims] = x[i].T
        new_x[i,  :, num_dims: 2*num_dims] = indicate_missingness_matrix[i].T
        new_x[i, :, 2*num_dims:] = count_between_missing_matrix[i].T

    x = new_x

    #Normalise
    num_features = np.shape(x)[2]
    for f in range(num_features):
        for day in range(num_steps):
            x[:, day, f] = x[:, day, f] / np.max(x[:,:, f] )

    
    #TCN parameters
    batch_size, timesteps, input_dim = None, num_steps, num_dims*3
    
    i1 = Input(batch_shape=(batch_size, timesteps, input_dim))
    t1 = TCN(return_sequences=False, dilations=[1, 2, 4, 8, 16, 32], dropout_rate = 0.1, use_skip_connections=True, padding='causal', use_layer_norm=True)(i1)
    ln1 = LayerNormalization()(t1)
    dr1 = Dropout(0.1)(ln1)

    max_neurons = 11

    total_val_acc = np.zeros((validation_runs, max_neurons, max_neurons))
    total_tr_acc = np.zeros((validation_runs, max_neurons, max_neurons))
    

    callback_loss = EarlyStopping(monitor='val_loss', min_delta= 0.01, patience=10, mode='auto', restore_best_weights=True)
    callback_acc = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=10, mode='auto', restore_best_weights=True)

    start = time.localtime()
    
    for num_neurons_de1 in range(2, max_neurons, 2):
        for num_neurons_de2 in range(2, max_neurons, 2):
            
            result_val_acc = np.zeros((validation_runs)) #mean of 10 last entries in history. TO DO: 2D by varying second dense layer
            result_tr_acc = np.zeros((validation_runs))
    
            if num_neurons_de2 <= num_neurons_de1:
                for i in range(validation_runs):
                
                    x_tr, y_tr = x[train_indices_array[i]], y[train_indices_array[i]]
                    x_te, y_te = x[test_indices_array[i]], y[test_indices_array[i]]

                    N_tr = len(y_tr)

                    class_weights = {0: 1, 1: (N_tr/sum(y_tr==1))}
                    
                    de1 = Dense(num_neurons_de1, activation='relu')(dr1)# to do: vary number of neurons
                    ln2 = LayerNormalization()(de1)
                    dr2 = Dropout(0.1)(ln2)
                    de2 = Dense(num_neurons_de2, activation='relu')(dr2)
                    ln3 = LayerNormalization()(de2)
                    dr3 = Dropout(0.1)(ln3)
                    o = Dense(1)(dr3)

                    m = Model(inputs=[i1], outputs=[o])
                    #keras.utils.plot_model(m, 'image.jpeg', show_shapes=True)#doesn't work, incompatibility issues
                    opt = keras.optimizers.Adam(learning_rate=0.0001)

                    m.compile(optimizer=opt, loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'] )
                    #m.compile(optimizer='adam', loss='hinge', metrics=['accuracy'] )

                    tcn_full_summary(m, expand_residual_blocks=False)

                    #Train, Validate
                    train_result = m.fit(x_tr, y_tr, epochs=250, validation_split=0.20, class_weight=class_weights, callbacks = [callback_loss, callback_acc])
                    result_val_acc[i] = np.max(train_result.history['val_accuracy'])
                    result_tr_acc[i] = train_result.history['accuracy'][np.argmax(train_result.history['val_accuracy'])]

                    total_val_acc[i, num_neurons_de1, num_neurons_de2] = result_val_acc[i]
                    total_tr_acc[i, num_neurons_de1, num_neurons_de2] = result_tr_acc[i]

            else:
                break
        

            plt.figure()

            plt.plot(result_val_acc, '-')
            plt.plot(result_tr_acc, '--')
            plt.xlabel('Run')
            plt.ylabel('Acc.')
            plt.grid('both')
            plt.legend(['Validation', 'Train'])
            plt.title('Validation')

            if not os.path.exists(figure_path + '{}'.format(method)):
                os.makedirs(figure_path + '{}/'.format(method))
            
            plt.savefig(figure_path + '{}/'.format(method) + 'tcn_architecture_val_and_train_accuracy_{}_cross_loss_candidate_{}_{}.png'.format(method, num_neurons_de1, num_neurons_de2), dpi=100)
            #plt.savefig(figure_path + 'tcn_architecture_val_and_train_accuracy_{}_hinge_loss.png'.format(method), dpi=100)
            #plt.show()
            plt.close()

    

    dense_layer_val_res_folder = ""

    if not synth:
        dense_layer_val_res_folder = folder + r'validation_of_dense_layers_candidate_for_{}\\'.format(method)
    else:
        dense_layer_val_res_folder = folder + r'validation_of_dense_layers_candidate_for_{}_synth\\'.format(method)
    
    if not os.path.exists(dense_layer_val_res_folder):
        os.makedirs(dense_layer_val_res_folder)

    for i in range(validation_runs):
        file_content_val = pd.DataFrame(total_val_acc[i]) 
        file_content_val.to_csv(dense_layer_val_res_folder + r'TCNresults_validation_{}_acc_cross_{}.txt'.format(i, method ), header=None, index=None, sep='\t', encoding='utf-8')

    end = time.localtime()
    delta_t = time.mktime(end) - time.mktime(start)
    frac_h = delta_t / (60*60)
    h = int(frac_h)
    frac_m = (frac_h-h) * 60
    mn = int(frac_m)
    frac_s = (frac_m - mn) * 60
    s = int(frac_s)
    print('Completion time: {} hours, {} minutes, {} seconds'.format(h, mn, s))


    plt.figure()
    plt.subplot(211)
    plt.imshow(np.median(total_val_acc, axis=0), interpolation='nearest', aspect='auto')
    plt.xlabel('Dense 2')
    plt.ylabel('Dense 1')
    plt.title('Validation Acc.')
    plt.xticks([i for i in range(max_neurons)])
    plt.yticks([i for i in range(max_neurons)])
    plt.colorbar()

    plt.subplot(212)
    plt.imshow(np.median(total_tr_acc, axis=0), interpolation='nearest', aspect='auto')
    plt.xlabel('Dense 2')
    plt.ylabel('Dense 1')
    plt.title(' Train Acc.')
    plt.xticks([i for i in range(max_neurons)])
    plt.yticks([i for i in range(max_neurons)])
    plt.colorbar()

    plt.subplots_adjust(hspace=0.47)
    plt.savefig(figure_path + 'tcn_architecture_val_and_train_accuracy_{}_cross_loss.png'.format(method), dpi=100)
    #plt.savefig(figure_path + 'tcn_architecture_val_and_train_accuracy_{}_hinge_loss.png'.format(method), dpi=100)
    #plt.show()
    plt.close()

    #Candidates: (dense 1: 10, dense 2: 2), (dense 1: 6, dense 2: 3), (dense 1: 9, dense 2: 9)

    # plt.figure()
    
    # plt.subplot(211)
    # plt.plot(train_result.history['loss'])
    # plt.plot(train_result.history['val_loss'])
    # plt.yscale('log')
    # plt.ylabel('Loss')
    # plt.xlabel('Epochs')
    # plt.grid('both')
    # plt.legend(['Train', 'Validation'])

    # plt.subplot(212)
    # plt.plot(train_result.history['accuracy'])
    # plt.plot(train_result.history['val_accuracy'])
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epochs')
    # plt.grid('both')
    # plt.suptitle('TCN classifier, {}-imputation'.format(method))
    # plt.legend(['Train', 'Validation'])
    
    # plt.subplots_adjust(bottom=1.1, top=1.2)
    # plt.savefig(figure_path + 'tcn_accuracy_{}.png'.format(method), dpi=100)
    # #plt.show()
    # plt.close()

    # #Test
    # y_pred = m.predict(x_te)
    # y_pred = y_pred > 0
    # #y_pred = np.argmax(y_pred >= 0.5, axis=1)

    # confusion_data = confusion_matrix(y_te, y_pred, labels=[0, 1])
    # confusion_dataframe = pd.DataFrame(confusion_data, index = ['True 0', 'True 1'], columns = ['Predicted 0', 'Predicted 1'])

    # #Test set results
    # balanced_acc_te, f1_score_te, accuracy_te = score_metrics(confusion_data)

    # # Accuracy and confusion table, DTW
    # print('TCN\n------')
    # print('Confusion matrix, test set:')
    # print(str( confusion_dataframe))
    # print('Accuracy: test = {}'.format(accuracy_te))
    # print('Balanced accuracy: test = {}'.format(balanced_acc_te))
    # print('F1: test = {}'.format(f1_score_te))
    # print('*************')

    # resultfile = open(folder + r'TCNresults.txt', 'a', encoding='utf-8')
    # resultfile.write('Confusion matrix, test set:\n')
    # resultfile.write(str( confusion_dataframe))
    # resultfile.write('\nAccuracy: test = {}\n'.format(accuracy_te))
    # resultfile.write('Balanced accuracy: test = {}\n'.format(balanced_acc_te))
    # resultfile.write('F1: test = {}\n'.format(f1_score_te))
    # resultfile.write('\n***************\n')
    # resultfile.close()

    # m.save(folder + 'tcn_model_{}'.format(method))
    # m.save_weights(folder + 'tcn_weights_{}'.format(method))
