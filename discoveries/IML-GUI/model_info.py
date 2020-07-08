# -*- coding: utf-8 -*-
"""
    Title   : Implementation of providing text to display on GUI
    @author : Sachin Nandakumar
"""

class Model_Info:
    '''
        This class defines methods that provides text about model details to display
        on the GUI. It also provide extra text to be printed in the tab4
    '''
    def get_info(self, model, status=None):
        '''
           This function gets the necessary information based on requirement 
        '''
        if model.lower() == 'gbt':
            info = self.get_info_xgb()
        else:
            info = self.get_info_svm()
        if status is not None:
            return info, self.param_range_instances(model, status)
        else:
            return info, None
            
    def get_info_xgb(self):
        '''
            This function outputs the text on GBT model details
        '''
        display_details ="<p><strong>Performance of model -&gt; GBT:</strong><br />======================<br />Accuracy: 0.65</p>"\
                         "<p><strong>Best Hyperparameter Settings:</strong><br />======================<br />n_estimators: 50<br />max_depth: 2<br />eta: 0.05<br />gamma: 0.5<br />colsample_bytree: 0.8<br />min_child_weight: 1<br />subsample: 1</p>"\
                         "<p><strong>Hyperparameter Grid:</strong><br />======================<br /><em>Unique values for xgboost</em><br />eta: [0.05, 0.1, 0.15, 0.2]<br />max_depth: [1, 2]<br />gamma: [0, 0.5, 1, 1.5, 2]<br />colsample_bytree: [0.8]<br />min_child_weight: [1]<br />subsample: [1]<br />n_estimators: [50, 100, 150, 200, 250, 300, 350]</p>"
        return display_details
    
    def get_info_svm(self):
        '''
            This function outputs the text on SVM model details
        '''
        display_details = "<p><strong>Performance of model -&gt; Linear SVM:</strong><br />======================<br />Accuracy: 0.64</p>"\
                          "<p><strong>Best Hyperparameter Settings:</strong><br />======================<br />Cost: 0.25</p>"\
                          "<p><strong>Hyperparameter Grid:</strong><br />======================<br /><em>Unique values for LinearSVC:</em><br />C: [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, 128]</p>"
        return display_details
        
    def param_range_instances(self, model, status):
        '''
            This function outputs the text on 'number of instances that satisfies the rule' for 
            both the models
        '''
        if model.lower() == 'gbt':
            if status == 1:
                display_details="<p><strong>Number of instances that </strong><strong>satisfy the rules</strong></p>\
                                <p>ar_2&nbsp; &nbsp; &nbsp; &nbsp; : 11<br />o_area_1 : 11<br />n_max&nbsp; &nbsp; &nbsp;: 11<br />sex_m&nbsp; &nbsp; &nbsp;: 16</p>"
            else:
                display_details="<p><strong>Number of instances that </strong><strong>satisfy the rules</strong></p>"\
                                "<p>n_avg&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; : 11<br />sex_m ^ a_area&nbsp; &nbsp; &nbsp; &nbsp;: 11<br />sex_m ^ ch_volume : 10<br />sex_f ^ n_avg&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; : 7</p>"
        else:
            if status == 1:
                display_details="<p><strong>Number of instances that </strong><strong>satisfy the rules</strong></p>"\
                                "<p>gamma&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; : 11<br />sex_m ^ h_max : 4</p>"
            else:
                display_details="<p><strong>Number of instances that </strong><strong>satisfy the rules</strong></p>"\
                                "<p>h_max : 11<br />ar_2&nbsp; &nbsp; : 11</p>"
        return display_details