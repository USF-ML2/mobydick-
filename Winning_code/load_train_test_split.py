"""
Code to load 30,000 AIFF files in python, sample them into
train & test & put them into 2 independent directories
"""

#Loading the libraries
import aifc
import pandas as pd
from sklearn import *


def aiff_train_test_split():
    #Setting the path & loading the data files
    path = '/Users/abhisheksingh29895/Dropbox/AdvancedML_Project_/Data/'
    all_train = []
    for i in range(30000):
        dat = aifc.open(path + 'train/'+'train'+str(i+1)+'.aiff','r')
        all_train.append(dat)

    #Loading the label files & taking the index
    labels = pd.read_csv(path+'train.csv')
    index = labels.icol(0).tolist()

    #Training & test split(33%) in Test set
    small_train, small_test, y_index, y_index = train_test_split(all_train,
                                                                 index,
                                                                 test_size=0.33, #33% in test set
                                                                 random_state=42) #Random generation

