# -*- coding: utf-8 -*-
#Comment: This code produces a graphic showing missingness as data, correalation, and ....

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
from vdr_imputation import Imputation
import os

X = loadmat('../SSI_data/X.mat')
x = X['X']

Y = loadmat('../SSI_data/Y.mat')
y = Y['Y']


num_days = 20

print(np.shape(y))

N = np.shape(y)[0] #total number of patients
test_names = ["Hemoglobin", "Leukocytes", "CRP", "Potassium", "Sodium", "Creatinine", "Thrombocytes", "Albumin", "Carbamide", "Glucose", "Amylase"]
num_tests = len(test_names) #number of tests

x = np.reshape(x, (N, num_tests, num_days))

time_index = np.array([i for i in range(num_days)])

for test_num in range(num_tests):
    for day in range(num_days):
        x[:, test_num, day] = x[:, test_num, day] / np.max(x[:, test_num, :])



missing = np.zeros((num_days,))
missing_rate = np.zeros((num_days,))
X_missing = np.zeros_like(y)
r = np.zeros((num_days,))



for k in range(num_tests):
    print('Missing:\tMissing rate:\tr:')
    for j in range(num_days):
        
        for i in range(N):
            X_missing[i] = np.sum( x[i, k, j] == 0 )

        # Math: Pearson Correlation
        a = N * sum(X_missing * y ) - sum(X_missing) * sum(y) 
        b = np.sqrt( N * sum(X_missing**2) - sum(X_missing)**2 ) * np.sqrt( N * sum(y**2) - sum(y)**2 )
        r[j] = a / b

        missing = np.sum( x[:, k, j] == 0 )
        missing_rate[j] = missing / N
    
        print('{:}\t\t{:.4f}\t\t{:.4f}'.format(missing, missing_rate[j], r[ j]))


    #Plot
    fig, (col_ax, ax1, ax2, ax3) = plt.subplots(nrows=4, gridspec_kw={'height_ratios':[0.1, 2, 0.5, 0.5]})
    
    img = ax1.imshow(x[:, k, :], interpolation='nearest', aspect='auto', cmap='PuRd')
    ax1.set_xticks(time_index)
    

    ax2.bar(time_index, r)
    ax2.set_ylabel('Pearson\nCorrelation')
    ax2.set_xticks(time_index)
    ax2.set_xlim(0, 19)
    ax2.set_ylim(-0.6, 0.2)
    ax2.grid(True, which='both', axis='y')
    ax2.tick_params(axis='y',
                    which='both',
                    left = False, 
                    right=False)

    ax3.bar(time_index, missing_rate)
    ax3.set_ylabel('Missing\nRate')
    ax3.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=True,      # ticks along the bottom edge are off
        top=True,         # ticks along the top edge are off
        labelbottom=True) # labels along the bottom edge are off
    ax3.set_xticks(time_index)
    ax3.set_xlim(0, 19)
    ax3.set_ylim(0, 1)
    ax3.grid(True, which='both', axis='y')
    ax3.set_xlabel('Day')

    fig.suptitle('{}'.format(test_names[k]))
    fig.colorbar(img, cax =col_ax, orientation='horizontal')
    fig.tight_layout()
    plt.savefig(r'..\\figures\\missingness\\inf_missingness_{}.png'.format(test_names[k]), dpi=100)




