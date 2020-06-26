# -*- coding: utf-8 -*-
"""
    Code for determination of value(s) and combination of features that is representative 
    of each class label by means of a Ruleset Model (RIPPER algorithm)
    
    @author: 
"""

import re
import pickle
import pandas as pd 
import wittgenstein as lw

class Parameter_Range:
    def __init__(self):
        dataset = pd.read_csv('data/Transformed_dataset.csv')
        scaler = pickle.load(open('models/data_transformation.pkl', 'rb'))
        self.X_transformed = dataset.iloc[:, :-1]
        self.y = dataset.iloc[:, -1]
        self.X = scaler.inverse_transform(self.X_transformed)
        self.X = pd.DataFrame(self.X, columns=list(self.X_transformed))
        

    def define_ruleset_model(self, model_name, status):
        '''
            This function predicts the classes based on RIPPER method
            for GBT or SVM model and creates corresponding model. 
            The model is then saved in to the directory for reuse.
        '''
        
        filename = 'models/model_'+model_name.lower()+'.pkl'
        model = pickle.load(open(filename, 'rb'))

        y_predicted = model.predict(self.X_transformed)
        
        clf = lw.RIPPER()
        clf.fit(self.X_transformed, y_predicted, pos_class=status, random_state=42)
        lol = clf.predict(self.X_transformed, give_reasons=True)
        count=1
        for each in lol[1]:
            if len(each)>0:
                count+=1
        print(clf.ruleset_.out_pretty())
        with open('models/ruleset_'+model_name.lower()+'_model_'+str(status)+'.pkl', 'wb') as mod:
            pickle.dump(clf, mod)
        
    
    def define_ruleset(self, model, status):
        '''
            This function returns the rulebook which consists of
            corresponding rules for the model and the status chosen
        '''
        rule_model = pickle.load(open('models/ruleset_'+model.lower()+'_model_'+str(status)+'.pkl', 'rb'))
#        cond_count = rule_model.ruleset_.count_conds()
        if status == 1:
            print("Rule set for class = Ruptured:")
        else:
             print("Rule set for class = Unruptured:")
        result = rule_model.ruleset_
        rule_book = {}
        for rule in result:
            if '^' in str(rule):
                rule_book = self.conjunctive_rules(str(rule), rule_book)
            else:
                rule_book = self.normal_rules(str(rule), rule_book)
        return rule_book
    
###################################################################################################################
        
    '''
        >>> Ruleset are disjunctions of several rules. 
            1. A rule in a ruleset can be conjunctions of rules, or
                Eg: (I’m good ^ we have enough time) 
            2. Simply a rule that defines some range of values it takes
                Eg: Age = 50-75
                
            To define both the scenarios (which occurs based on the model and the dataset), 
                1. conjunctive_rules()
                2. normal_rules()
    '''
    
    def conjunctive_rules(self, rules, rule_book):
        '''
            This function formats values and standards of conjunctive rules and writes it to rulebook
        '''
        print('inside conjunctive_rules')
        rulebook_key = ''
        rulebook_values = []
        for each in rules.split('^'):
            rule, val = self.rule_format(each)
            rulebook_values.append(val[0])
            if not rules.split('^')[-1] == each:
                rulebook_key += rule + '^'
            else:
                rulebook_key += rule
        rule_book[rulebook_key] = rulebook_values
        return rule_book
    
    
    def normal_rules(self, rule, rule_book):
        '''
            This function formats values and standards of normal rules and writes it to rulebook
        '''
        print('inside normal_rules')
        rule, rulebook_values = self.rule_format(rule)
        rule_book[rule] = rulebook_values
        return rule_book
        
    def rule_format(self, each):
        '''
            This method formats normal/conjunctive rules with corresponding values
        '''
        rule = str(each).replace(']','').replace('[','').split('=')
        rulebook_values = []
        if rule[1].count('-') == 0:
            rulebook_values.append(self.inverse_transformation(rule[0], [rule[1]]))
        elif rule[1].count('-') == 1:
            if rule[1].startswith('-'):
                rulebook_values.append(self.inverse_transformation(rule[0], [rule[1]]))
            else:
                rulebook_values.append(self.inverse_transformation(rule[0], rule[1].split('-')))
        elif rule[1].count('-') == 2:
            if rule[1].startswith('-'):
                rulebook_values.append(self.inverse_transformation(rule[0], rule[1].rsplit('-', 1)))
        else:
            rulebook_values.append(self.inverse_transformation(rule[0], re.findall(r"[-+]?\d*\.\d+|\d+", rule[1])))
        
        return rule[0], rulebook_values
        

    def inverse_transformation(self, rule, list_rule_values):
        '''
            This function makes sure that the final result returned are inverse_transformed 
            to its original set of values from the z-score transformed dataset
        '''
        transformed_list = []
        for each in list_rule_values:
            val = float(each)*self.X[[rule]].std() + self.X[[rule]].mean()
            transformed_list.append(round(val[0], 2))
        return transformed_list