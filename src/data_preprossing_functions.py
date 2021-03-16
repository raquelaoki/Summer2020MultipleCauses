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

def clinical_filtering(clinical):
  #Filter 1: Remove CIO_Grade Exclude and 1
  print('Original shape: ', clinical.shape)
  clinical_ = clinical[clinical.CIO_Grade != 'Exclude']
  clinical_ = clinical_[clinical_.CIO_Grade != 1]
  print('# Filter 1: new shape: ', clinical_.shape)

  #Create Y: CIO grade: 0 -> 0, 2-4 -> 1 (hearing loss happened)
  #clinical_ = clinical_.astype({'CIO_Grade': 'int32'})
  clinical_['y']=[0 if item == 1 else 1 for item in clinical_.CIO_Grade]
  #fillna columns with NAN values (from 0 days or 0 doses)
  clinical_.fillna(0, inplace = True)

  #x days -> x
  columns_mix_numbers_string = ['VancomycinConcomitantDuration (days)','TobramycinConcomitantDuration (days)',
                                'GentamicinConcomitantDuration (days)','AmikacinConcomitantDuration (days)',
                                'FurosemideConcomitantDuration (days)']
  for column in columns_mix_numbers_string:
    clinical_[column] = ['0 days' if item == 0 else item for item in clinical_[column]]
    clinical_[column] = clinical_[column].str.extract('(\d+)')

  #categorical variables to binary
  clinical_['Otoprotectant_Given'] = [0 if item == 'Not given' else 1 for item in clinical_['Otoprotectant_Given']]
  clinical_['Array'] = [0 if item == 'Omni' else 1 for item in clinical_['Array']]

  #remove any remain nans
  clinical_ = clinical_.dropna()
  print('# Filter 2: final shape', clinical_.shape)
  return clinical_

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


def run_preprocessing(path, save_files = False):
    #Loading files
    try:
        file = 'clinical_01Mar2021.xlsx'
        clinical = pd.read_excel(file) #new data set
    except FileNotFoundError:
        print('ERROR: ',file, ' missing from Colab')

    #Working only with dominant encoding
    path, tag = "/content/", 'snpsback_'

    try:
        data_df = sparse.load_npz(path+tag+'gt_dominant.npz') #gt following snps and variants order
        variants_f = np.load(path + tag+'variants.npy') #variants names
        cadd_f = np.load(path+ tag+ 'cadd.npy') #cadd score following the variants order
        samples_f = np.load(path+ 'samples.npy',allow_pickle = True ) #samples order
        ks = pd.read_csv(path + 'known_snps_fullname.txt', header = None)[0].values #known snps
    except FileNotFoundError:
        print('ERRROR: gt_dominant,samples, cadd, variants or known_snps_fullname missing from Colab')
    #From Data pre-processing functions:
    clinical_ = clinical_filtering(clinical)
    data_df_, samples_f = update_data_after_clinical_filtering(data_df, samples_f, clinical_)
    clinical_['ID'] = pd.Categorical(clinical_['ID'], categories=samples_f,   ordered=True)
    clinical_ = clinical_.sort_values(by=['ID'])
    clinical_.reset_index(inplace=True, drop=True)

    ks_f = ks_bool(ks,variants_f)
    data_df_, variants_f, cadd_f, ks_f = eliminate_low_incidence(data_df_, variants_f, cadd_f, ks_f)
    data_df_, variants_f, cadd_f, ks_f = eliminate_low_cadd(data_df_, variants_f, cadd_f, ks_f)
    y = clinical_['y']
    clinical_ = clinical_.drop(["CIO_Grade",'y'], axis = 1)
    if save_files:
        np.save('y',y)
        np.save('x_clinical', clinical_)
        np.save('x_snps', data_df_)
        np.save('x_colnames', variants_f)

    return y,clinical_ , data_df_, variants_f
