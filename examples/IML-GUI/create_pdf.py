# -*- coding: utf-8 -*-
"""
    Title   : Implementation of creating and formatting PDF for download of reports
    @author : Sachin Nandakumar
"""

import os
import numpy as np
import pandas as pd
from statistics import mean
import matplotlib.pyplot as plt
from fpdf import FPDF, HTMLMixin
from generate_ice import Generate_ICE
from model_reliance import Model_Reliance
from param_range import Parameter_Range

import warnings
warnings.filterwarnings("ignore")

class MyFPDF(FPDF, HTMLMixin):
    pass

class Create_PDF:
    
    def __init__(self):
        self.pdf = MyFPDF()
        self.pdf.compress = False
        self.pdf.add_page()
        self.pdf.set_font('Arial', '', 14)  
        self.pdf.ln(10)
        
    
    def save_plots(self):
        '''
            This function directs to the corresponding tasks to save the plots to 
            the directory inorder to populate it to PDF 
        '''
        model_list = ['gbt','svm']  
        parameter_list = ['a_area', 'a_volume', 'alpha', 'ar_1', 'ar_2', 'beta', 'ch_area', 
                          'ch_volume', 'd_max', 'delta_ab', 'ei','gamma', 'h_max', 'h_ortho', 
                          'n_avg', 'n_max', 'nsi', 'o_area_1', 'o_area_2', 'ui', 'w_max', 'w_ortho']
        self.gc = Generate_ICE()
        self.model_reliance = Model_Reliance()
        self.param_range = Parameter_Range()
        for model in model_list:
            self.draw_model_reliance(model)
            for status in [0,1]:
                self.print_parameter_range(model, status)
            for parameter in parameter_list:
                self.display_ICE_plots(model, parameter)
                
    def draw_model_reliance(self, model):
        '''
            This function saves the task1 bar chart to the directory
        '''
        if not os.path.isdir('img/model_reliance_{}.jpg'.format(model)): 
            data = self.model_reliance.model_reliance(model)
            figure = plt.figure(figsize=(8,6))
            figure.clear()
            ax = figure.add_subplot(111)
            figure.suptitle('Feature Importance - Model Reliance', fontsize=20)
            ax.set_xlabel('Importance', fontsize=18)
            ax.set_ylabel('Features', fontsize=18)
            ind = np.arange(len(list(data.values())))
            ax.barh(ind, list(data.values()), tick_label=list(data), label='features')
            figure.savefig('img/model_reliance_{}.jpg'.format(model))
            plt.clf()
    
    def display_ICE_plots(self, model, parameter):
        '''
            This function saves the task2 ICE plot to the directory
        '''
        if not os.path.isdir('img/ice_plot_{}_{}.jpg'.format(model, parameter)): 
            test, test_label, ice_feature_col, z_temp = self.gc.generate_ICE(model, parameter)
            for ICE_type in [0, 1]:
                j = rug = 0
                k = len(ice_feature_col)
                figure = plt.figure(figsize=(8,6))
                figure.clear()
                k = len(ice_feature_col)
                pdp_temp = [0]*len(ice_feature_col) #For having a PDP line
                ax = figure.add_subplot(111)
                ax.plot([], color='tab:blue', label ='non-ruptured')  
                ax.plot([], color='tab:orange',label = 'ruptured')
                
                #Create Temporary dataframe for generating ICE plots
                for i in range(int(len(test)/len(ice_feature_col))):
                    temp5 = pd.concat([z_temp[j:k],test[j:k]],axis=1)
                    label_temp = float(test_label[j:k].mean()[0]) #For splitting lines into two labels
                    temp5.columns = ('{}'.format(parameter),'probability')
                    temp5_1 = temp5.sort_values(by=['{}'.format(parameter)])
                    rug_temp = 0
                    center_val = 0
                    
                    #ICE plots
                    if ICE_type == 1:
                        if label_temp < 0.5:
                            ax.plot(temp5_1['{}'.format(parameter)],temp5_1['probability'],color = 'tab:blue',lw = 0.7,label='_nolegend_')
                        else:
                            ax.plot(temp5_1['{}'.format(parameter)],temp5_1['probability'],color = 'tab:orange', lw = 0.7,label='_nolegend_')
                        pdp_arr = list(temp5_1['probability'])
                        pdp_temp = [x + y for x, y in zip(pdp_temp , pdp_arr)]
                        rug_temp = float(temp5_1['{}'.format(parameter)].min())
                        
                    #Centered ICE plots
                    else:            
                        center_val = temp5_1.iloc[0,1] #Centered according to the smallest value 
                        temp6 = pd.DataFrame(temp5_1['probability'] - center_val)
                        if label_temp < 0.5:
                            ax.plot(temp5_1['{}'.format(parameter)],temp6,color = 'tab:blue',lw = 0.7,label='_nolegend_')
                        else:
                            ax.plot(temp5_1['{}'.format(parameter)],temp6,color = 'tab:orange', lw = 0.7,label='_nolegend_')
                        rug_temp = float(temp6.min())
                    
                    j+=len(ice_feature_col)
                    k+=len(ice_feature_col)
                    
                    
                    if rug_temp < rug:
                        rug = rug_temp
                        
                if ICE_type == 1:        
                    pdp_temp[:] = [g / 100 for g in pdp_temp]   
                    plt.plot(temp5_1['{}'.format(parameter)],pdp_temp,color = 'tab:purple',lw = 5,label='_nolegend_')
                        
                        
                '''
                    Add rugs dynamically according to the least value on y-axis(probability) predicted by each
                    feature. The three conditions take care of positioning of the rugs so the plots don't get messy.
                '''
                if rug < 0: 
                    ax.plot(temp5_1['{}'.format(parameter)], [rug*1.1]*len(temp5_1), '|', color='k')
                elif rug == 0:
                    rug = -0.001
                    ax.plot(temp5_1['{}'.format(parameter)], [rug]*len(temp5_1), '|', color='k')
                else:
                    ax.plot(temp5_1['{}'.format(parameter)], [rug*0.1]*len(temp5_1), '|', color='k')
                
                ax.set_xlabel('{}'.format(parameter), fontsize=18)
                ax.set_ylabel('Predicted rupture probability', fontsize=18)
                ax.legend(loc = "upper right")
                if ICE_type == 1:
                    figure.savefig('img/iceplot_{}_{}_{}.jpg'.format(model, parameter, ICE_type), dpi=100) #Remove if you don't want to save fig 
                else:
                    figure.savefig('img/iceplot_{}_{}_{}.jpg'.format(model, parameter, ICE_type), dpi=100)
                plt.clf()
#                ax = figure.add_subplot(111)
#                for i in range(int(len(test)/len(ice_feature_col))):
#                    temp5 = pd.concat([z_temp[j:k],test[j:k]],axis=1)
#                    temp5.columns = ('{}'.format(parameter),'probability')
#                    temp5_1 = temp5.sort_values(by=['{}'.format(parameter)])
#                    rug_temp = 0
#                    #normal
#                    if ICE_type == 1:
#                        ax.plot(temp5_1['{}'.format(parameter)],temp5_1['probability'])
#                        rug_temp = float(temp5_1['{}'.format(parameter)].min())
#                    else:             #centered
#                        center_val = temp5_1.iloc[0,1]
#                        temp6 = pd.DataFrame(temp5_1['probability'] - center_val)
#                        plt.plot(temp5_1['{}'.format(parameter)],temp6)
#                        rug_temp = float(temp6.min())
#                    
#                    j+=len(ice_feature_col)
#                    k+=len(ice_feature_col)
#                    
#                
#                    if rug_temp < rug:
#                        rug = rug_temp
#        
#                if rug < 0:
#                    ax.plot(temp5_1['{}'.format(parameter)], [rug*1.1]*len(temp5_1), '|', color='k')
#                elif rug == 0:
#                    rug = -0.001
#                    ax.plot(temp5_1['{}'.format(parameter)], [rug]*len(temp5_1), '|', color='k')
#                else:
#                    ax.plot(temp5_1['{}'.format(parameter)], [rug*0.1]*len(temp5_1), '|', color='k')
#            
#                ax.set_xlabel('{}'.format(parameter), fontsize=18)
#                ax.set_ylabel('Predicted rupture probability', fontsize=18)
#                if ICE_type == 1:
#                    figure.savefig('img/iceplot_{}_{}_{}.jpg'.format(model, parameter, ICE_type), dpi=100) #Remove if you don't want to save fig 
#                else:
#                    figure.savefig('img/iceplot_{}_{}_{}.jpg'.format(model, parameter, ICE_type), dpi=100)
#                plt.clf() 
    
    def print_parameter_range(self, model, status):
        '''
            This function saves the task4 box plot to the directory
        '''
        if not os.path.isdir('img/box_plot_{}_{}.jpg'.format(model,status)):
            figure = plt.figure(figsize=(8,6))
            figure.clear()
            data = self.param_range.define_ruleset(model, status)
            ax = figure.add_subplot(111)
            figure.suptitle('Range of Parameters', fontsize=20)
            ax.set_xlabel('Parameters', fontsize=18)
            ax.set_ylabel('Range of values', fontsize=18)
            pos, color_count = 1, 0
            x_ticks = []
            colors = ['blue', 'lightgreen', 'red', 'cyan', 'yellow', 'orange']
            for k, v in data.items():
                bp = ax.boxplot(v, positions = list(range(pos, pos+len(v))), widths = 0.6, patch_artist=True)
                for box in bp['boxes']:
                    # change outline color
                    box.set(color='black', linewidth=2)
                    # change fill color
                    box.set(facecolor = colors[color_count] )
                color_count += 1
                    
                x_ticks.append(mean(list(range(pos, pos+len(v)))))
                pos += len(v) + 2
            
            ax.set_xticklabels(data.keys())
            for i in range(0, len(x_ticks)):
                if not x_ticks[i] == x_ticks[-1]:
                    ax.axvline(x=mean([x_ticks[i],x_ticks[i+1]]))
            ax.set_xticks(x_ticks)
            figure.savefig('img/box_plot_{}_{}.jpg'.format(model, str(status)))
            plt.clf()
        
    def insert_task2_gbt(self): 
        '''
            Insert ICE plots for GBT model to the PDF in proper side-by-side format
        '''
        prefixed = [filename for filename in os.listdir('./img') if filename.startswith("iceplot_gbt")]
        for i in range(0, len(prefixed), 2):
            if prefixed[i] != 'iceplot_gbt_age_0.jpg':
                if prefixed[i] is not prefixed[-1]:
                    if self.pdf.get_y() >= 225:
                        self.pdf.add_page()
                    self.pdf.image('img/{}'.format(prefixed[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
                    self.pdf.image('img/{}'.format(prefixed[i+1]), x = 110, y = self.pdf.get_y(), w = 90, h = 60)
                else:
                    self.pdf.image('img/{}'.format(prefixed[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
                self.pdf.set_y(self.pdf.get_y()+60)

    def insert_task2_svm(self): 
        '''
            Insert ICE plots for SVM model to the PDF in proper side-by-side format
        '''
        prefixed = [filename for filename in os.listdir('./img') if filename.startswith("iceplot_svm")]
        for i in range(0, len(prefixed), 2):
            if prefixed[i] != 'iceplot_svm_age_0.jpg':
                if prefixed[i] is not prefixed[-1]:
                    if self.pdf.get_y() >= 225:
                        self.pdf.add_page()
                    self.pdf.image('img/{}'.format(prefixed[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
                    self.pdf.image('img/{}'.format(prefixed[i+1]), x = 110, y = self.pdf.get_y(), w = 90, h = 60)
                else:
                    self.pdf.image('img/{}'.format(prefixed[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
                self.pdf.set_y(self.pdf.get_y()+60)
    
    def insert_task3_gbt(self): 
        '''
            Insert Density plots for GBT model to PDF in proper side-by-side format
        '''
        prefixed = [filename for filename in os.listdir('./img') if filename.startswith("counterfactual_gbt")]
        split_prefixed = []
        for files in prefixed: 
            instance = files.split('_')[2]+'_'
            if instance not in split_prefixed:
                split_prefixed.append(instance)
        for instance_no in split_prefixed:
            instance_CFs = [filename for filename in prefixed if instance_no in filename]
            for i in range(0, len(instance_CFs), 2):
                if instance_CFs[i] is not instance_CFs[-1]:
                    if self.pdf.get_y() >= 225:
                        self.pdf.add_page()
                    self.pdf.image('img/{}'.format(instance_CFs[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
                    self.pdf.image('img/{}'.format(instance_CFs[i+1]), x = 110, y = self.pdf.get_y(), w = 90, h = 60)
                else:
                    self.pdf.image('img/{}'.format(instance_CFs[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
                self.pdf.set_y(self.pdf.get_y()+60)
            self.pdf.set_y(self.pdf.get_y()+40)

                    
    def insert_task3_svm(self): 
        '''
            Insert Density plots for SVM model to PDF in proper side-by-side format
        '''
        prefixed = [filename for filename in os.listdir('./img') if filename.startswith("counterfactual_svm")]
        split_prefixed = []
        for files in prefixed: 
            instance = files.split('_')[2]+'_'
            if instance not in split_prefixed:
                split_prefixed.append(instance)
        for instance_no in split_prefixed:
            instance_CFs = [filename for filename in prefixed if instance_no in filename]
            for i in range(0, len(instance_CFs), 2):
                if instance_CFs[i] is not instance_CFs[-1]:
                    if self.pdf.get_y() >= 225:
                        self.pdf.add_page()
                    self.pdf.image('img/{}'.format(instance_CFs[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
                    self.pdf.image('img/{}'.format(instance_CFs[i+1]), x = 110, y = self.pdf.get_y(), w = 90, h = 60)
                else:
                    self.pdf.image('img/{}'.format(instance_CFs[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
                self.pdf.set_y(self.pdf.get_y()+60)
            self.pdf.set_y(self.pdf.get_y()+40)
    
    def insert_task4_gbt(self): 
        '''
            Insert box plots for GBT model to PDF in proper side-by-side format
        '''
        prefixed = [filename for filename in os.listdir('./img') if filename.startswith("box_plot_gbt")]
        for i in range(0, len(prefixed), 2):
            if prefixed[i] is not prefixed[-1]:
                if self.pdf.get_y() >= 225:
                    self.pdf.add_page()
                self.pdf.image('img/{}'.format(prefixed[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
                self.pdf.image('img/{}'.format(prefixed[i+1]), x = 110, y = self.pdf.get_y(), w = 90, h = 60)
            else:
                self.pdf.image('img/{}'.format(prefixed[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
            self.pdf.set_y(self.pdf.get_y()+60)
    
    def insert_task4_svm(self): 
        '''
            Insert box plots for SVM model to PDF in proper side-by-side format
        '''
        prefixed = [filename for filename in os.listdir('./img') if filename.startswith("box_plot_svm")]
        for i in range(0, len(prefixed), 2):
            if prefixed[i] is not prefixed[-1]:
                if self.pdf.get_y() >= 225:
                    self.pdf.add_page()
                self.pdf.image('img/{}'.format(prefixed[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
                self.pdf.image('img/{}'.format(prefixed[i+1]), x = 110, y = self.pdf.get_y(), w = 90, h = 60)
            else:
                self.pdf.image('img/{}'.format(prefixed[i]), x = 10, y = self.pdf.get_y(), w = 90, h = 60)
            self.pdf.set_y(self.pdf.get_y()+60)
    
    
    def html2pdf(self, filename):
        '''
            This function prints text and images to the PDF based on set alignment        
        '''
        self.save_plots()
        if not filename.endswith('.pdf'):
            filename += '.pdf'
            
        html = ('<h2 align="center">Interpretable Machine Learning for Rupture Risk<br />Classification in Intracranial Aneurysms</h2>'+
                '<h3 align="center">Result Report</h3>'+
                '<p align="center">________________________________________________________________________</p>'+
                '<p>This report generated from the GUI consists of information of the dataset that was used for the study, the classification models that&nbsp;were considered, different areas explored within the project in terms of interpretable machine learning and the plots and figures that&nbsp;justify the same.</p>'+
                '<p>&nbsp;</p>'+
                '<h3>The Dataset</h3>'+
                '<p >The dataset consists of demographical and morphological parameters of 74 patients with 100 intracranial aneurysms which were developed at the University Hospital of Magdeburg, Germany.</p>'+
                '<p>&nbsp;</p>'+
                '<h3>Classification Models</h3>'+
                '<p >For the project implementation, the team came up with 2 models particularly:</p>'+
                '<p>    1. Gradient Boosted Trees: We used XGBoost which is an optimized distributed gradient boosting library based on the Gradient Boosting framework.</p>'+
                '<p>    2. Linear SVM: In particular, in terms of implementation, we used Linear Support Vector Classification (LinearSVC) library of sklearn tool.</p>'
                )
        
        html2 = ('<p>&nbsp;</p><h3> 1. Global Feature Importance</h3>'+
                 '<p>We find the importance of a feature based on the increase in the prediction error of the model using a model-agnostic version of feature importance called model reliance based on Fisher, Rudin, Dominici (2018).</p></br>'
                 '<p>                                        GBT                                                                                  SVM</p>'
                 )
        
        html3 = ('<h3> 2. Individual Conditional Expectation (ICE) plots for feature relationship</h3>'+
                 '<p>Determines how the instance prediction changes when a feature changes. For this very task, we used individual conditional expectation (ICE) plot (Goldstein et al. 2017). An ICE plot visualizes the dependence of the prediction on a feature for each instance separately, resulting in one line per instance, compared to one line overall in partial dependence plots.</p>'+
                 '<p>&nbsp;</p><p>        GBT                Centered ICE                                                                  Normal ICE</p>'
                )
        
        html4 = ('<p>&nbsp;</p><p>        SVM                Centered ICE                                                                  Normal ICE</p>'
                )
        
        html5 = ('<h3> 3. Density plots(Counterfactual) of feature(s) of instances</h3>'+
                 '<p>Determines counterfactuals for all possible features of an instance which is visualized using Density Plots</p>'
                 '<p>Since saving and displaying all the features are quite time and space consuming, the plots for instance 52 will be provided by default. For any other instance, the desired combination of \'model\' and \'instance\' should be selected in the GUI. If counterfactuals for the instance exists, the plot will be displayed in this file.</p>'
                )
        
        html6 = ('<h3> 4. RIPPER method for determining parameter value range</h3>'+
                 '<p>The next task of the project was to determine range and combination of parameters that are predicted to contribute towards specific type of classification by satisfying majority number of instances. This is visualized using boxplots that defines the range of (combination of) parameters</p>'+
                 '<p>&nbsp;</p><p>        GBT                Status: Unruptured                                                                  Status: Ruptured</p>'
                 )
        
        html7 = ('<p>&nbsp;</p><p>        SVM                Status: Unruptured                                                                  Status: Ruptured</p>'
                )
        
        self.pdf.write_html(html)
        self.pdf.image('commons/model_details.JPG', x = 60, y = 160, w = 90, h = 45)
        self.pdf.set_xy(10, 200)
        self.pdf.write_html(html2)
        self.pdf.image('img/model_reliance_gbt.JPG', x = 10, y = 235, w = 90, h = 60)
        self.pdf.image('img/model_reliance_svm.JPG', x = 110, y = 235, w = 90, h = 60)
        self.pdf.add_page()
        self.pdf.write_html(html3)
        self.pdf.set_y(55)
        self.insert_task2_gbt()
        self.pdf.write_html(html4)
        self.pdf.set_y(205)
        self.insert_task2_svm()
        self.pdf.set_xy(10, self.pdf.get_y()+10)
        self.pdf.add_page()
        self.pdf.write_html(html5)
        self.pdf.set_xy(10, self.pdf.get_y()+10)
        self.insert_task3_gbt()
        self.pdf.set_xy(10, self.pdf.get_y()+10)
        self.pdf.add_page()
        self.pdf.write_html(html6)
        self.pdf.set_xy(10, self.pdf.get_y()+10)
        self.insert_task4_gbt()
        self.pdf.write_html(html7)
        self.pdf.set_xy(10, self.pdf.get_y()+10)
        self.insert_task4_svm()
        self.pdf.output(filename, 'F')