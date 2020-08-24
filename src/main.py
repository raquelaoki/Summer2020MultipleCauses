# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 09:37:31 2020

@author: raoki
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import sparse
from os import listdir
from os.path import isfile, join
import data_preprossing_functions as dpp


path = 'C://Users//raque//Documents//GitHub//Summer2020MultipleCauses'
sys.path.append(path+'//src')
sys.path.append(path+'//parkca')
import train as parkca
import eval as evaluation

os.chdir(path)

DAbool = True
BARTCEVAEbool = True
RUN_learners = False
RUN_meta = True
RUN_dataprep = False


#path_input = "Z:\\POPi3\\Staff\\Raquel Aoki\\Cisplatin-induced_ototoxicity"
#path_output = "Z:\\POPi3\\Staff\\Raquel Aoki\\Summer2020\\"
path_data= "C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\"
path_output= "C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\results"
known_snps_path = 'C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\known_snps_fullname.txt'

if RUN_dataprep:
    tag = 'snpsback_'
    data_df = sparse.load_npz(path_data+tag+'gt_dominant.npz')
    data_sf = sparse.load_npz(path_data+tag+'gt_recessive.npz')
    variants_f = np.load(path_data+tag + 'variants.npy')
    cadd_f = np.load(path_data+tag+'cadd.npy')
    samples_f = np.load(path_data+ 'samples.npy',allow_pickle = True )
    
    ks = pd.read_csv(known_snps_path, header = None)[0].values#Not All are inthe variants, some are missing
    
    '''CLINICAL INFORMATION'''
    clinical = dpp.clinical_filtering(path_data+'clinical_data.csv')
    #update data_df with new rows
    data_df_, samples_f = dpp.update_data_after_clinical_filtering(data_df, samples_f, clinical)
    
    clinical['ID'] = pd.Categorical(clinical['ID'], categories=samples_f,   ordered=True)
    clinical = clinical.sort_values(by=['ID'])
    clinical.reset_index(inplace=True, drop=True)
    
    
    '''Data Prep'''
    ks_f = dpp.ks_bool(ks,variants_f)
    data_df_, variants_f, cadd_f, ks_f = dpp.eliminate_low_incidence(data_df_, variants_f, cadd_f, ks_f)
    data_df_, variants_f, cadd_f, ks_f = dpp.eliminate_low_cadd(data_df_, variants_f, cadd_f, ks_f)

'''lEARNER 1: THE DECONFOUNDER'''
#the function learners don't work directly, proposing a new one here:


if RUN_learners: 
    X = data_df_
    clinical = clinical.astype({'age': 'float','cisp_dur':'float'})
    Z = clinical.drop(['ID','y'],axis=1)
    colnamesZ = Z.columns
    Z = Z.values
    y = clinical.y.values
    
    level1data = parkca.learners_bcch(path_output, DAbool, BARTCEVAEbool, X, y, variants_f,'snps', Z,colnamesZ)
    level1data['y_out'] = 0 # based on ks
    level1data['y_out'] = [1 if i in ks else 0 for i in level1data.snps]
    level1data.set_index('snps', inplace = True, drop = True)
    qav, q_ = evaluation.diversity(['cevae'],['coef'], level1data)
    print('DIVERSITY: ', qav)
    level1data.to_csv(path_output+'\\level1data.txt', sep=';')

if RUN_meta: 
    level1data = pd.read_csv(path_output+'\\level1data.txt', sep=';')
    level1data.set_index('snps', inplace = True, drop = True) 
    data1 = parkca.data_norm(level1data)
    experiments1 = parkca.meta_learner(data1.drop(['coef'],axis =1), ['rf'],1)
    experiments1.to_csv(path_output+'\\eval_metalevel1.txt', sep=';')


'''


#Metalearners
experiments0 = eval.first_level_asmeta(['bart_all',  'bart_FEMALE',  'bart_MALE' ],
                    ['dappcalr_15_LGG','dappcalr_15_SKCM','dappcalr_15_all','dappcalr_15_FEMALE','dappcalr_15_MALE'],
                    data1)

experiments1.to_csv('results\\eval_metalevel1.txt', sep=';')
experiments1c.to_csv('results\\eval_metalevel1c.txt', sep=';')
experiments0.to_csv('results\\eval_metalevel0.txt', sep=';')
print("DONE WITH EXPERIMENTS ON APPLICATION")

'''


