"""
    Code for generating regular ICE and centered-ICE plots
    @author: 
"""

import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings("ignore")


class Generate_ICE:
    def __init__(self):
        self.anurysm_df = pd.read_csv('data/Transformed_dataset.csv')
        self.anurysm_df.drop('ruptured',1,inplace = True) #drop label column from main Df
        self.z_score = pd.read_csv('data/Transformed_dataset_orig.csv')

    def generate_ICE(self, model, ice_feature):
        
        with open('models/model_'+model+'.pkl', 'rb') as f:  #Load pickle model
            data = pickle.load(f)
        
        col_names = self.anurysm_df.columns.values.tolist()
        
        encoding_dataframe = pd.DataFrame(self.anurysm_df)
    
     
        '''Ensure that feature for ICE is not categorical''' 
    
#        if any(c in ice_feature for c in ("side","sex_f","sex_m","type_BF","type_SW")):
#            print("\nCannot perform ICE plot for selected column because of categorical format. PDP should be done instead.")
#            continue
        
        ice_feature_col = encoding_dataframe[ice_feature].to_frame() #Separate the ICE Attribute
            
        training_df  = pd.DataFrame(encoding_dataframe) #Create Another dataframe making ICE data
        encoding_dataframe.drop(ice_feature,1,inplace = True) #Drop the ice attribute
           
    
        ''' We need column location of the ice feature for later use where we will reorder the columns 
        for created ice data according to the data we are training with the model or else it will be 
        extremely horrible prediction!'''
     
        location = training_df.columns.get_loc(ice_feature)
    
    
        #Select all unique values out of the selected column
        selected_feature = ice_feature_col[ice_feature].unique()
    
        #Make ICE data for other features keeping them as it is!
        other_features = pd.DataFrame(encoding_dataframe)
    
        temp2 = other_features.loc[np.repeat(other_features.index.values,selected_feature.size)].reset_index()
        temp2 = temp2.drop('index', axis=1)
        
        #Replicate the selected feature's data as of rest of ICE data
        selected_feature = pd.DataFrame(selected_feature)
        temp3 = pd.concat([selected_feature] * len(other_features.index.values),ignore_index=True)
        
        '''Make another one of that selected feature ICE column with original values(before z-score transformed)'''
        
        z_feature = pd.DataFrame(self.z_score[ice_feature].unique())
        z_temp = pd.concat([z_feature] * len(other_features.index.values),ignore_index=True)
       
        #Join the created ICE Data
        temp4 = pd.concat([temp3,temp2],1)
        temp4.set_axis(1, range(len(temp4.columns)))  #Reset-Index of columns
    
    
        ''' Use the location value here to reorder columns for final prediction based on the saved trained model'''
        ice_train_df = temp4    
        cols = list(ice_train_df)   
        cols.insert(location, cols.pop(cols.index(0)))
        ice_train_df = ice_train_df.ix[:, cols]
        ice_train_df.set_axis(1, range(len(ice_train_df.columns)))
    
        ''' Specifically for using pickle model, use the column names as it is!'''
        ice_train_df.columns = col_names
        
        '''The order of the classes when predicting probability corresponds to that in the attribute classes_ .
        (see official sklearn documentation)'''
        
        test = data.predict_proba(ice_train_df)   
        test_label = pd.DataFrame(data.predict(ice_train_df))
        test = pd.DataFrame(test)
        test = test[1]
        
       
        return test, test_label, ice_feature_col, z_temp