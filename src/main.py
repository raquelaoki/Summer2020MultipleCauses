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

#path_input = "Z:\\POPi3\\Staff\\Raquel Aoki\\Cisplatin-induced_ototoxicity"
#path_output = "Z:\\POPi3\\Staff\\Raquel Aoki\\Summer2020\\"
path_data= "C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\"
path_output= "C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\results"
known_snps_path = 'C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\known_snps_fullname.txt'

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
DAbool = True
BARTbool = False
X = data_df_

clinical = clinical.astype({'age': 'float','cisp_dur':'float'})
Z = clinical.drop(['ID','y'],axis=1)
colnamesZ = Z.columns
Z = Z.values
y = clinical.y.values

#parkca.learners_bcch(path_output, DAbool, BARTbool, X, Z, y, variants_f, colnamesZ, 'snps')


coef, coef_continuos, roc, coln = parkca.deconfounder_PPCA_LR(X,variants_f,y,'DA',15,100,Z,colnamesZ)
roc_table = pd.DataFrame(columns=['learners', 'fpr','tpr','auc'])
roc_table = roc_table.append(roc,ignore_index=True)
evaluation.roc_plot(roc_table)

