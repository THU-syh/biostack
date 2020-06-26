"""
    Code for determination of counterfactual features of instances
    @author: 
"""


import numpy as np
import pandas as pd 
import pickle
import math
import seaborn as sns
import matplotlib.pyplot as plt


class Counterfactuals:
    def __init__(self):
        self.df = pd.read_csv('data/Transformed_dataset.csv')
        self.z_score = pd.read_csv('data/Transformed_dataset_orig.csv')
        Y = self.df.iloc[:,-1]
        self.y = np.array(Y)
        self.X = self.df.drop('ruptured', axis =1)
        self.col_names = self.X.columns.values.tolist()
        self.x = np.array(self.X) 

    def getCFList(self, minVal, maxVal,instVal):
        '''
            This function will return list of all the counterfactuals in pre-selected range.
            The returned List will be sorted according to the distance from the original value
            
            params:
                 minVal     : minimum value of the feature colum
                 maxVal     : maximum value of the feature column
                 instVal    : feature value- to be used to sort the CF values
                 
            Output: 
            CFList has two values: 
                    1. Difference between actual instance and new found counterfactual value(so 
                       that we find the nearest counterfactual - if we sort it in ascending order)
                    2. Actual counterfactual value
        '''
       
        bins = 100
        feature_range = maxVal - minVal
        feature_change= feature_range/bins
        CFList = pd.DataFrame(columns = ['diff', 'val'])
        for i in range(bins):
            new_CF_val = minVal + (i*feature_change)
            CFList.loc[i,'diff']= abs(instVal - new_CF_val)
            CFList.loc[i,'val'] = new_CF_val
        List = CFList.sort_values(by = 'diff')
        return List
    
    def getBestCF(self, List,CF_instance,colcnt,des_outcome, data):
        ''' 
            This function returns best counterfactual for a feature. Since the CFList is already sorted based on the 
            distance from the original value, the first Counterfactual instance returned will be the best(nearest) one.
        '''
        for i in range(len(List)):
            updated_val = List.loc[i,'val']
            CF_instance[0,colcnt] = updated_val
            CF_Inst_1 = pd.DataFrame(CF_instance)
            CF_Inst_1.columns = self.col_names
            pred_inst_counterfactual = data.predict(CF_Inst_1)
            if(pred_inst_counterfactual == des_outcome):
                return updated_val
    
    def gen_kde(self, kde_feature, feature_name, instance_no, model):
        '''
            This function generate density plots with location of original as well as counterfactual values
        '''
        
        # For inverse transforming Z-score
        z_values = []
        z_values.append([float(self.z_score[feature_name].mean()),float(self.z_score[feature_name].std())])
        
        #Define the original value(Get location), the counterfactual value(And Inverse it!)
        densi_val = kde_feature
        inverse_densi_val = densi_val*z_values[0][1]+z_values[0][0]
        
        col_loc = self.z_score.columns.get_loc(feature_name)
        
        original_val = self.z_score.iloc[instance_no,col_loc] #Get orignal value from instance no and column loc
        
        # split data into ruptured and unruptured
        rup , non_rup = [x for _,x in self.z_score[feature_name].groupby(self.df.iloc[:,-1] < 1)]

        # generate kernel density estimation plot
        plt.figure(figsize=(8,6), dpi=100)
        plt.suptitle('Density Plot(Counterfactual) for {} of {}'.format(feature_name, instance_no), fontsize=20)
        
        # plot the lines depicting those values
        p = sns.kdeplot(rup,shade = True, label = 'ruptured', c = 'tab:orange')
        sns.kdeplot(non_rup,shade = True, label = 'non-ruptured', c = 'tab:blue')   
        
        plt.axvline(inverse_densi_val,0,0.5,ls = '--', c = 'tab:blue', label = 'counterfactual instance')
        plt.axvline(original_val,0,0.7,ls = '--', c = 'tab:orange', label = 'original instance')
        p.annotate('{:.2f}'.format(inverse_densi_val),xy =(inverse_densi_val,0))
        p.annotate('{:.2f}'.format(original_val),xy =(original_val,0),horizontalalignment='right')
        plt.ylabel('Density')
        plt.xlabel(feature_name)
        plt.legend()
        plt.savefig("img/counterfactual_{}_{}_{}.jpg".format(model.lower(), instance_no, feature_name))
        plt.clf()
    


    def get_best_counterfactuals(self, instance, model): 
        inst_interest = pd.DataFrame(self.x[[instance]])
        
        with open('models/model_{}.pkl'.format(model), 'rb') as f:  #Load pickle model
                data = pickle.load(f)
        
        # run model to predict outcome for instance of interest
        inst_interest.columns = self.col_names
        pred_outcome = data.predict(inst_interest)

        #find out desired outcome then change the label accordingly to find alternate outcome
        des_outcome = 1 if pred_outcome == 0 else 0
        colcnt=0
        CF_Inst = inst_interest
        kde_feat_list = []
        for feature_col in self.X.columns:
            CF_Inst = self.x[[instance]]
            feature_min = min(self.X[feature_col])
            feature_max = max(self.X[feature_col])
            # get Counterfactual values for the feature_col
            CFList_feature = self.getCFList(feature_min,feature_max,CF_Inst[0,colcnt])
            #best_CF: get best counterfactual for each feature.
            upd_val = self.getBestCF(CFList_feature,CF_Inst,colcnt,des_outcome, data)
            # show these Counterfactuals in a table
            CF_Inst[0,27] = upd_val
            
            # Generate Density plots for features where counterfactual is found
            if not math.isnan(CF_Inst[0,27]):
                kde_feat_list.append(feature_col)
                self.gen_kde(CF_Inst[0,colcnt], feature_col, instance, model)
                
            #hardcoded value restricted to number of features of the given data-set 
            if(colcnt<28):
                colcnt = colcnt+1
            else:
                break;
        return kde_feat_list
