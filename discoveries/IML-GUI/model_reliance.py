# -*- coding: utf-8 -*-
"""
    Title   : Implementation of determining the feature importance using Model Reliance
    @author : Sachin Nandakumar
"""

import collections
import operator
import pickle
import pandas as pd

class Model_Reliance:
    def __init__(self):
        dataset = pd.read_csv('data/Transformed_dataset.csv')
        self.X = dataset.iloc[:, :-1]
        self.y = dataset.iloc[:, -1]
        
    def permute_dataset(self, X):
        '''
            Permute the dataset by shuffling the first and second half of each columns with one another 
        '''
        X_perm1, X_perm2 = X.iloc[:50, :].set_index([pd.Index(range(50, 100))]), X.iloc[50:, :].reset_index()
        return pd.concat([X_perm2, X_perm1])
    
    def find_error(self, y_true, y_pred):
        '''
            Find the misclassification error by comparing original and estimated target classes
        '''
        count = 0
        for i in range(0, len(y_true)):
            if y_true[i] != y_pred[i]:
                count += 1
        return count
    
    def model_reliance(self, model):
        '''
            Takes in the desired model for which the target classes for all instances are predicted and the
            original model error are found out. 
            
            params:
                model: the model for which the permutation feature importance is to be calculated
                
            output:
                permutation_feature_importance: a dictionary of {features:values} sorted in descending order
        '''
        filename = 'models/model_'+model.lower()+'.pkl'
        
        model = pickle.load(open(filename, 'rb'))
        x_predicted = model.predict(self.X)
        
        e_org = self.find_error(self.y, x_predicted)
        
        # permute the whole dataset
        permuted_X = self.permute_dataset(self.X)
        permutation_feature_importance = {}
        for feature in list(self.X):
            temp = self.X.copy()
            # replace the original values of a feature (column) with permuted values 
            # keeping rest of the values in the original dataset the same
            temp[feature] = permuted_X[feature]
            temp_predicted = model.predict(temp)                # predict the classes with this locally altered dataset
            e_perm = self.find_error(self.y, temp_predicted)    # find the permuted error
            permutation_feature_importance[feature] = e_perm - e_org    #find the difference between two errors as permutation feature importance
        permutation_feature_importance = collections.OrderedDict(permutation_feature_importance)
        permutation_feature_importance = dict(
                sorted(permutation_feature_importance.items(), key=operator.itemgetter(1)))
        return permutation_feature_importance