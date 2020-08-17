# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 09:37:31 2020

@author: raoki
"""
import os
import numpy as np 
import pandas as pd 
from scipy import sparse

path_input = "Z:\\POPi3\\Staff\\Raquel Aoki\\Cisplatin-induced_ototoxicity"
path_output = "Z:\\POPi3\\Staff\\Raquel Aoki\\Summer2020\\"
path_output_simulated = "C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\"
known_snps_path = 'C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\known_snps.txt'

tag = 'snpsback_'
data_df = sparse.load_npz(path_output+tag+'gt_dominant.npz')
data_sf = sparse.load_npz(path_output+tag+'gt_recessive.npz')
variants_f = np.load(path_output+tag + 'variants.npy')
cadd_f = np.load(path_output+tag+'cadd.npy')
ks = pd.read_csv(known_snps_path, header = None)[0].values#Not All are inthe variants, some are missing

