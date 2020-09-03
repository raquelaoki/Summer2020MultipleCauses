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

randseed = 123
print("random seed: ", randseed)
np.random.seed(randseed)

DAbool = False
BARTCEVAEbool = False
RUN_learners = False
RUN_meta = True
RUN_dataprep = True


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
    level1data = level1data.sample(frac=1).reset_index(drop=True)
    level1data.set_index('snps', inplace = True, drop = True) 
    data1 = parkca.data_norm(level1data)
    experiments1,predictions, pred_pro = parkca.meta_learner(data1.drop(['coef'],axis =1), ['rf'],1)
    experiments1.to_csv(path_output+'\\eval_metalevel1.txt', sep=';')
    print(experiments1.transpose())
    
    #from sklearn.metrics import confusion_matrix,f1_score
    #print(confusion_matrix(predictions['rf'],data1['y_out']))
    #tn, fp, fn, tp
    
    pred_causes = []
    snps= []
    prob_cause = []
    missing_known = []
    confirm_known = []
    for i in range(len(predictions['rf'])): 
        if predictions['rf'][i]==1: 
            snps.append(data1.index[i])
            prob_cause.append(pred_pro[i,1])
            if data1['y_out'][i] == 1: 
                confirm_known.append(data1.index[i])
                    
        elif predictions['rf'][i]==0 and data1['y_out'][i] == 1: 
            missing_known.append(data1.index[i])
            snps.append(data1.index[i])
            prob_cause.append(pred_pro[i,1])
    
    
    legend = []
    
    for i in range(len(snps)): 
        if snps[i] in missing_known: 
            legend.append('Missed')
        elif snps[i] in confirm_known: 
            legend.append('Confirmed')
        else: 
            legend.append('Potential')
     
    results = pd.DataFrame({'snps':snps,'legend':legend,'prob':prob_cause})
    results.to_csv(path_output+'\\results_snps.csv')

    print(ks)

'''Outcome Prediction using the causal features'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.metrics import confusion_matrix,f1_score,accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

def predict_outcome(y_train, y_test, X_train, X_test):    
    w = len(y_train)/y_train.sum()
    sample_weight = np.array([w if i == 1 else 1 for i in y_train])
    model = RandomForestClassifier(max_depth=6, random_state=0)
    model.fit(X_train, y_train,sample_weight = sample_weight)
    
    y_train_p = model.predict(X_train)
    y_test_p = model.predict(X_test)
    print('Random Forest\n')
    aux = precision_recall_fscore_support(y_train_p, y_train)
    aux_ = precision_recall_fscore_support(y_test_p, y_test)
    output = [accuracy_score(y_train_p, y_train), aux[0][0],aux[1][0],aux[0][1],aux[1][1]]
    output_ = [accuracy_score(y_test_p, y_test),aux_[0][0],aux_[1][0],aux_[0][1],aux_[1][1]]
    #print(output,'\n',confusion_matrix(y_train_p, y_train),'\n',)
    #print(output_,'\n', confusion_matrix(y_test_p, y_test),'\n')
    return output, output_
    

#Option 1: using the 300
#option 2: using the != potential (known)
#option 3: using the confirmedis (-potential - missing)
#option 4: using only clinical information (sex, ..., ) 
#option 5: Random
ks = pd.read_csv(path_output+'\\results_snps.csv')

def predictive_models_comp(option,ks,data_df_,variants_f,clinical):
    if option ==  2:
        print(option)
        ks = ks[ks.legend!='Potential']
        print(len(ks))
    elif option == 3: 
        print(option)
        ks = ks[ks.legend=='Confirmed']
    else: 
        print(option)


    remove = []
    for i in range(len(variants_f)): 
        if variants_f[i] not in ks.snps.values: 
            remove.append(i)
     
    if option ==5:
        remove = np.random.choice(np.arange(len(variants_f)),40)
   
    X = np.delete(data_df_,remove, axis =1)
    colX = np.delete(variants_f, remove, axis = 0 ) 
    
    y = clinical['y'].values
    Z = clinical.drop(['ID','y','cisp_dose','cisp_dur','carb'],axis = 1)
    scaler = MinMaxScaler()
    Z = scaler.fit_transform(Z)
    
    if option == 4:
        X = Z
    else:
        X = np.concatenate([Z,X], axis = 1)
        
    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.3,random_state=3)
    o, o_ = predict_outcome(y_train, y_test, X_train, X_test)
    roc = {'type':option,'train':o, 'test':o_}
    return roc
   
    
roc_table = pd.DataFrame(columns=['type','train','test'])  
roc_table= roc_table.append(predictive_models_comp(1,ks,data_df_,variants_f,clinical), ignore_index=True)
roc_table = roc_table.append(predictive_models_comp(2,ks,data_df_,variants_f,clinical), ignore_index=True)
roc_table= roc_table.append(predictive_models_comp(3,ks,data_df_,variants_f,clinical), ignore_index=True)
roc_table= roc_table.append(predictive_models_comp(4,ks,data_df_,variants_f,clinical), ignore_index=True)
roc_table = roc_table.append(predictive_models_comp(5,ks,data_df_,variants_f,clinical), ignore_index=True)
  

key = ['All SNPS','Known SNPS','Confirmed SNPS','Clinical','Random']
for i in range(roc_table.shape[0]):
    print('\n\nModel',key[i],':\nTest:', roc_table.test[i],'\nTrain:',roc_table.train[i])


  