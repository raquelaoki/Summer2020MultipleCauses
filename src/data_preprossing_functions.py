# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 09:50:25 2020

@author: raoki
"""
from scipy import sparse 
import numpy as np 
import pandas as pd 

def ks_bool(ks,variants_f):
    #Create an bool array with ks positions
    #For some reasons there is a repeated element in ks : 67
    ks_f = [1 if v in ks else 0 for v in variants_f]
    ks_f  = list(map(bool,ks_f))
    ks_f = np.array(ks_f)
    return ks_f

def clinical_filtering(path): 
    clinical = pd.read_csv(path,encoding = "ISO-8859-1")
    print('Original Clinical Info Shape:', clinical.shape)
    #1) Remove CIO GRADE 2 and Exclude
    clinical_ = clinical[clinical.CIO_Grade!='2']
    clinical_ = clinical_[clinical_.CIO_Grade!='Exclude']
    if clinical_['CIO_Grade'].isnull().values.any(): 
        print('There are NAN values')
    
    #2) CIO grade: 0 and 1 -> 0, 2 ignore, 4-3 -> 1 (hearing loss happened)
    clinical_ = clinical_.astype({'CIO_Grade': 'int32'})
    clinical_['y']=[0 if item <= 1 else 1 for item in clinical_.CIO_Grade]
    
    #2) Remove Carbo = 1
    #clinical_ = clinical_[clinical_['Carboplatin (1=yes,0=no)']==0]
    #3) Remove cancer type columns for now (too little samples to be relevante)
    #4) Keep: 'ID',  'Sex (1=male, 0=female)', 'AgeTreatmentInitiation (years)', 
    #'CisplatinDose (mg/m2)', 'CisplatinDuration (days)','Carboplatin (1=yes,0=no)',  'y'
    keep = ['ID',  'Sex (1=male, 0=female)', 'AgeTreatmentInitiation (years)', 
            'CisplatinDose (mg/m2)', 'CisplatinDuration (days)',
            'Carboplatin (1=yes,0=no)',  'y']
    clinical_1 = clinical_[keep]
    clinical_1=clinical_1.rename(columns ={'Sex (1=male, 0=female)':'sex', 'AgeTreatmentInitiation (years)':'age', 
            'CisplatinDose (mg/m2)':'cisp_dose', 'CisplatinDuration (days)':'cisp_dur',
            'Carboplatin (1=yes,0=no)':'carb'})
    print('New Clinical Info Shape:', clinical_1.shape)
    print(clinical_1.head())
    return clinical_1

def update_data_after_clinical_filtering(data_df,samples_f, clinical): 
    remove = []
    for i in range(len(samples_f)):
        if samples_f[i] not in clinical.ID.values: 
            remove.append(i)
    
    data_df_= data_df.todense().transpose()
    data_df_ = np.delete(data_df_, remove, axis = 0)
    samples_f = np.delete(samples_f, remove, axis=0)
    return data_df_, samples_f



def eliminate_low_incidence(data_df, variants_f, cadd_f, ks_f):
    #Remove columns whose less than 1% of counts are 1 in the data_df
    #Remove columns whose less than 1% of counts are 0 in the data_sf 
    print('\n\nFiltering Low incidence')
    sum_df = np.array(data_df.sum(axis = 0))
    th = np.ceil(data_df.shape[0]*0.05)
    sum_df_greater = np.greater_equal(sum_df,th) #SNPS less then th are False
    print('Before adding Known SNPS: ', sum_df_greater.sum())
    sum_df_greater = np.add(sum_df_greater[0,:], ks_f)
    print('After adding Known SNPS: ', sum(sum_df_greater))
    
    remove = []
    for i in range(len(sum_df_greater)): 
        if not sum_df_greater[i]: 
            remove.append(i)
    
    data_df_ = data_df#.todense().transpose()
    print('Original Shape Before Filtering: Data', data_df_.shape,', variants ' ,variants_f.shape,
          ', Cadd', cadd_f.shape, ', Ks',ks_f.shape)
    data_df_ = np.delete(data_df_,remove, axis =1)
    variants_f = np.delete(variants_f, remove, axis= 0)
    cadd_f = np.delete(cadd_f, remove, axis= 0)
    ks_f = np.delete(ks_f, remove, axis= 0)
    print('New Shape After Filtering: Data', data_df_.shape,', variants ' ,variants_f.shape,
          ', Cadd', cadd_f.shape, ', Ks',ks_f.shape)
    return data_df_, variants_f, cadd_f, ks_f



def eliminate_low_cadd(data_df, variants_f, cadd_f, ks_f):
    #Remove columns whose CADD less than 10 
    print('\n\nFiltering Low Cadd')
    sum_df_greater = np.greater_equal(cadd_f,10) #SNPS less then th are False
    print('Before adding Known SNPS: ', sum_df_greater.sum())
    sum_df_greater = np.add(sum_df_greater, ks_f)
    print('After adding Known SNPS: ', sum(sum_df_greater))
    
    remove = []
    for i in range(len(sum_df_greater)): 
        if not sum_df_greater[i]: 
            remove.append(i)
    
    data_df_ = data_df#.todense().transpose()
    print('Original Shape Before Filtering: Data', data_df_.shape,', variants ' ,variants_f.shape,
          ', Cadd', cadd_f.shape, ', Ks',ks_f.shape)
    data_df_ = np.delete(data_df_,remove, axis =1)
    variants_f = np.delete(variants_f, remove, axis= 0)
    cadd_f = np.delete(cadd_f, remove, axis= 0)
    ks_f = np.delete(ks_f, remove, axis= 0)
    print('New Shape After Filtering: Data', data_df_.shape,', variants ' ,variants_f.shape,
          ', Cadd', cadd_f.shape, ', Ks',ks_f.shape)
    return data_df_, variants_f, cadd_f, ks_f
