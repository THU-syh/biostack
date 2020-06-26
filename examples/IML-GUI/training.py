# -*- coding: utf-8 -*-
"""
    Title   : Code for preprocessing of the Aneurysm dataset
    @author : Sachin Nandakumar
"""


from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split

from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import make_scorer
import time
from os import path

def create_dataset(filename):
    '''
        This function gets the filename of CSV file (dataset) and converts it to 
        dataframe. And then converted to input dataframe and output vector (target class)
    '''
    dataset = pd.read_csv(filename)
    
    X = dataset.iloc[:, 1:28]
    X = X.drop(['location', 'ch_area', 'ch_volume', 'ar_2', 'ar_1', 'n_avg', 'n_max', 'w_ortho', 'h_ortho', 'h_max', 'd_max', 'o_area_1', 'a_volume', 'a_area', 'side'] ,1)
    y = dataset.iloc[:, 0]
    return X, y


def data_impute(X, fill_zeros=True, univariate_mean=False, univariate_mostfreq=False, knn=False, k=4):
    '''
        Data Imputation Techniques
            1. Filling with zeroes - Default
            2. Filling with Mean
            3. Filling with Most-Frequent
            4. k-Nearest Neighbours
    '''
    
    if fill_zeros:
        columns = X.columns
        X = X.fillna(0)
    
    if univariate_mean or univariate_mostfreq:
        from sklearn.impute import SimpleImputer
        if univariate_mean:
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        else:
            imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
        columns = X.columns
        X = imp.fit_transform(X)
    
    if knn:
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(max_iter=k, random_state=0, initial_strategy = 'most_frequent')
        columns = X.columns
        X = imp.fit_transform(X)
    
    X = pd.DataFrame(X, columns=columns)
    return X


def cat_encode(X):
    '''
        Category Encoding Techniques
        
        this part of the code is a general snippet which 
        identifies categorical features in 'any' dataset and
        and encodes it accordingly.
        i.e., cat_features = total_features - numerical_features
    '''
    
    # get all numerical columns in the dataset
    num_cols = X.select_dtypes([np.number]).columns
    # extract out all categorical features from the dataset
    encoded_x = None
    encoded_feature_names = []
    
    # cat_features = total_features - numerical_features
    cat_features = list(set(list(X.columns)).symmetric_difference(set(list(num_cols))))
    
    # cat_features = ['sex','type']
    print(cat_features)
    for i in cat_features:
        feature = X[i].values.reshape(X.shape[0], 1)
        onehot_encoder = OneHotEncoder(sparse=False)
        
        #   here, feature will have 3 columns: for example, for gender/sex, after onehotencoding,
        #   it gets converted to Male, Female and Nans.
        #   if data_imputation(X) =+ fill_zeros, then feature[feature == 0] = 'nan'
        #   else, feature[feature == 2] = 'nan'
        unique, _ = np.unique(feature.astype(str), return_counts=True)
        feature = onehot_encoder.fit_transform(feature.astype(str))
        
        #   Remove Nan column.
            # if data_imputation(X) == fill_zeros
        feature = feature[:, 1:]
            # if data_imputation(X) == univariate/knn
#        feature = feature[:, :-1]           
    
    
        #   Concatenating all encoded feature columns together
        if encoded_x is None:
            encoded_x = feature
        else:
            encoded_x = np.concatenate((encoded_x, feature), axis=1)
    
        # giving an understandable column names to new encoded column names
        # for eg: sex will have 2 unique values -> m & f
        print('unique:', unique)
        
        # renaming them to sex_m & sex_f
            # if data_imputation(X) == fill_zeros
        unique = list(i + '_' + unique[j] for j in range(1, len(unique)))
#             if data_imputation(X) == univariate/knn
#        unique = list(i + '_' + unique[j] for j in range(0, len(unique)-1))
    
        for i in range(0, len(unique)):
            encoded_feature_names.append(unique[i])
    
        print('encoded_feature_names: ',encoded_feature_names)
    
    # encoded_x: columns with values but no column names (headers)
    # encoded_feature_names: column names
    encoded_df = pd.DataFrame(encoded_x, columns=encoded_feature_names)
    # drop those categorical features (sex, type) from the main dataframe
    X = X.drop(cat_features, axis=1)
    # concat new encoded columns (of sex & type) to the dataframe
    X = pd.concat([X, encoded_df], axis=1)
    final_columns = list(X)
    return X, final_columns


def data_transformation(X, final_columns, norm=False, z_score=True):
    '''
        Data transformation techniques
            1. Range transformation (Normalization)
            2. Z-Score transformation (Standardization) - Default
        
    '''
    X_transformed = X
    #   necessary transformations
    if norm:
        norm = Normalizer()
        X_transformed = norm.fit_transform(X)
        X_transformed = pd.DataFrame(X_transformed, columns=final_columns)
        print('Normalized')
    if z_score:
        scaler = StandardScaler()
        X_transformed = scaler.fit_transform(X)
        X_transformed = pd.DataFrame(X_transformed, columns=final_columns)
        print('Z-Score Applied')
        
        print(X_transformed)
        X_transformed_inversed = pd.DataFrame(scaler.inverse_transform(X_transformed), columns=final_columns)
        print(X_transformed_inversed)
        
        fi = 'data_transformation.pkl'
        with open(fi, 'wb') as mod:
            pickle.dump(scaler, mod)    

    return X_transformed

def nested_cv(X_transformed, y, tuned_parameters, model=XGBClassifier()):
    '''
        This function finds generalization performance of the model using inner cross-validation
        technique. 
        params:
                X_transformed   : input dataset (transformed)
                y               : target class
                tuned_parameters: set of hyperparameters for which the model has to be evaluated
                model           : model for classification
                
        output: generalization performance of the model
    '''

    sss_outer = StratifiedShuffleSplit(n_splits=10, test_size=0.4, random_state=15)
    sss_inner = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=16)
    
    clf = GridSearchCV(estimator=model, param_grid=tuned_parameters, n_jobs=-1, verbose=1, scoring='accuracy', cv=sss_inner)
    nested_score = cross_validate(clf, X_transformed, y, cv=sss_outer)

    print('\n >>>The Generalization Performance is:')
    for name in nested_score.keys():
        print('%s: %.3f (+/-%0.03f)' % (name, np.mean(nested_score[name]), np.std(nested_score[name])))


def cross_vali(X_transformed, y, tuned_parameters, model=XGBClassifier()):
    '''
        This function does cross-validation on the dataset to determine the best set of hyperparameters
        Performs repeated stratified cross-validation
        params:
                X_transformed   : input dataset (transformed)
                y               : target class
                tuned_parameters: set of hyperparameters for which the model has to be evaluated
                model           : model for classification
                
        output: tuned model
    '''
    kFold = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=53)
    start_time = time.time()
    print('Start time normalCV:', time.ctime(start_time))
    clf = GridSearchCV(XGBClassifier(), tuned_parameters, n_jobs=-1, verbose=1, cv=kFold, scoring='accuracy', iid=False)
    clf.fit(X_transformed, y)
    end_time = time.time()
    print("time taken for normalCV: %s seconds ---" % (end_time - start_time))
    print(clf.best_estimator_)
    print("Best: %f using %s" % (clf.best_score_, clf.best_params_))
    return clf

def save_model(model, filepath):
    '''
        This function saves the model as pickle file in to the directory
    '''
    with open(filepath, 'wb') as mod:
        pickle.dump(model, mod)
    
    if not path.exists('data/Transformed_dataset.csv'):
        df_tosave = pd.concat([X_transformed, y], axis=1)
        df_tosave.to_csv('data/Transformed_dataset.csv', index=False)
   
    
filename = "data/aneur.csv"
X, y = create_dataset(filename)
print('>>>>>>>Created Dataframe')

X = data_impute(X)
print('>>>>>>>Data Imputation Done')

X, final_columns = cat_encode(X)
print('>>>>>>>Encoded Categorical Data')
print(list(X))
X = X.drop(['type_SW', 'type_BF'], 1)
final_columns.remove('type_SW')
final_columns.remove('type_BF')
print(final_columns)
X_transformed = data_transformation(X, final_columns, False, True)
print('>>>>>>>Transformed Data')

#gradientBoostedClassifier
#tuned_parameters = [{'learning_rate': np.linspace(0.05, 0.2, 4, endpoint=True), 'max_depth': [1, 2], 'n_estimators':range(50, 400, 50), 
#                     'subsample': [1]}]

#xgboost
#tuned_parameters = [{'eta': np.linspace(0.05, 0.2, 4, endpoint=True), 'max_depth': [1, 2], 
#                       'n_estimators':range(50, 400, 50), 'min_child_weight': [1], 'gamma': np.linspace(0, 2, 5, endpoint=True), 
#                         'colsample_bytree': [0.8], 'subsample': [1]}]
    
# LinearSVC
tuned_parameters = [{'C': [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]}]

    
nested_cv(X_transformed, y, tuned_parameters, LinearSVC())

#model = cross_vali(X_transformed, y, tuned_parameters, LinearSVC())
    
#print('>>>>>>>Selection of Best HyperParameters done')

#save_model(model, "models/model_svm.pkl")
#print('>>>>>>>Model Saved!!!')