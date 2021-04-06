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
sys.path.append(path + '//src')
sys.path.append(path + '//parkca')
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

# path_input = "Z:\\POPi3\\Staff\\Raquel Aoki\\Cisplatin-induced_ototoxicity"
# path_output = "Z:\\POPi3\\Staff\\Raquel Aoki\\Summer2020\\"
path_data = "C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\"
path_output = "C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\results"
known_snps_path = 'C:\\Users\\raque\\Documents\\GitHub\\Summer2020MultipleCauses\\data\\known_snps_fullname.txt'

if RUN_dataprep:
    tag = 'snpsback_'
    data_df = sparse.load_npz(path_data + tag + 'gt_dominant.npz')
    data_sf = sparse.load_npz(path_data + tag + 'gt_recessive.npz')
    variants_f = np.load(path_data + tag + 'variants.npy')
    cadd_f = np.load(path_data + tag + 'cadd.npy')
    samples_f = np.load(path_data + 'samples.npy', allow_pickle=True)

    ks = pd.read_csv(known_snps_path, header=None)[0].values  # Not All are inthe variants, some are missing

    '''CLINICAL INFORMATION'''
    clinical = dpp.clinical_filtering(path_data + 'clinical_data.csv')
    # update data_df with new rows
    data_df_, samples_f = dpp.update_data_after_clinical_filtering(data_df, samples_f, clinical)

    clinical['ID'] = pd.Categorical(clinical['ID'], categories=samples_f, ordered=True)
    clinical = clinical.sort_values(by=['ID'])
    clinical.reset_index(inplace=True, drop=True)

    '''Data Prep'''
    ks_f = dpp.ks_bool(ks, variants_f)
    data_df_, variants_f, cadd_f, ks_f = dpp.eliminate_low_incidence(data_df_, variants_f, cadd_f, ks_f)
    data_df_, variants_f, cadd_f, ks_f = dpp.eliminate_low_cadd(data_df_, variants_f, cadd_f, ks_f)

'''lEARNER 1: THE DECONFOUNDER'''
# the function learners don't work directly, proposing a new one here:


if RUN_learners:
    X = data_df_
    clinical = clinical.astype({'age': 'float', 'cisp_dur': 'float'})
    Z = clinical.drop(['ID', 'y'], axis=1)
    colnamesZ = Z.columns
    Z = Z.values
    y = clinical.y.values

    level1data = parkca.learners_bcch(path_output, DAbool, BARTCEVAEbool, X, y, variants_f, 'snps', Z, colnamesZ)
    level1data['y_out'] = 0  # based on ks
    level1data['y_out'] = [1 if i in ks else 0 for i in level1data.snps]
    level1data.set_index('snps', inplace=True, drop=True)
    qav, q_ = evaluation.diversity(['cevae'], ['coef'], level1data)
    print('DIVERSITY: ', qav)
    level1data.to_csv(path_output + '\\level1data.txt', sep=';')

if RUN_meta:
    level1data = pd.read_csv(path_output + '\\level1data.txt', sep=';')
    level1data = level1data.sample(frac=1).reset_index(drop=True)
    level1data.set_index('snps', inplace=True, drop=True)
    data1 = parkca.data_norm(level1data)
    experiments1, predictions, pred_pro = parkca.meta_learner(data1.drop(['coef'], axis=1), ['rf'], 1)
    experiments1.to_csv(path_output + '\\eval_metalevel1.txt', sep=';')
    print(experiments1.transpose())

    # from sklearn.metrics import confusion_matrix,f1_score
    # print(confusion_matrix(predictions['rf'],data1['y_out']))
    # tn, fp, fn, tp

    pred_causes = []
    snps = []
    prob_cause = []
    missing_known = []
    confirm_known = []
    for i in range(len(predictions['rf'])):
        if predictions['rf'][i] == 1:
            snps.append(data1.index[i])
            prob_cause.append(pred_pro[i, 1])
            if data1['y_out'][i] == 1:
                confirm_known.append(data1.index[i])

        elif predictions['rf'][i] == 0 and data1['y_out'][i] == 1:
            missing_known.append(data1.index[i])
            snps.append(data1.index[i])
            prob_cause.append(pred_pro[i, 1])

    legend = []

    for i in range(len(snps)):
        if snps[i] in missing_known:
            legend.append('Missed')
        elif snps[i] in confirm_known:
            legend.append('Confirmed')
        else:
            legend.append('Potential')

    results = pd.DataFrame({'snps': snps, 'legend': legend, 'prob': prob_cause})
    results.to_csv(path_output + '\\results_snps.csv')

    print(ks)

'''Outcome Prediction using the causal features'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support


def predict_outcome(y_train, y_test, X_train, X_test):
    w = len(y_train) / y_train.sum()
    sample_weight = np.array([w if i == 1 else 1 for i in y_train])
    model = RandomForestClassifier(max_depth=6, random_state=0)
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_train_p = model.predict(X_train)
    y_test_p = model.predict(X_test)
    print('Random Forest\n')
    aux = precision_recall_fscore_support(y_train_p, y_train)
    aux_ = precision_recall_fscore_support(y_test_p, y_test)
    output = [accuracy_score(y_train_p, y_train), aux[0][0], aux[1][0], aux[0][1], aux[1][1]]
    output_ = [accuracy_score(y_test_p, y_test), aux_[0][0], aux_[1][0], aux_[0][1], aux_[1][1]]
    # print(output,'\n',confusion_matrix(y_train_p, y_train),'\n',)
    # print(output_,'\n', confusion_matrix(y_test_p, y_test),'\n')
    return output, output_


# Option 1: using the 300
# option 2: using the != potential (known)
# option 3: using the confirmedis (-potential - missing)
# option 4: using only clinical information (sex, ..., )
# option 5: Random
ks = pd.read_csv(path_output + '\\results_snps.csv')


def predictive_models_comp(option, ks, data_df_, variants_f, clinical):
    if option == 2:
        print(option)
        ks = ks[ks.legend != 'Potential']
        print(len(ks))
    elif option == 3:
        print(option)
        ks = ks[ks.legend == 'Confirmed']
    else:
        print(option)

    remove = []
    for i in range(len(variants_f)):
        if variants_f[i] not in ks.snps.values:
            remove.append(i)

    if option == 5:
        remove = np.random.choice(np.arange(len(variants_f)), 40)

    X = np.delete(data_df_, remove, axis=1)
    colX = np.delete(variants_f, remove, axis=0)

    y = clinical['y'].values
    Z = clinical.drop(['ID', 'y', 'cisp_dose', 'cisp_dur', 'carb'], axis=1)
    scaler = MinMaxScaler()
    Z = scaler.fit_transform(Z)

    if option == 4:
        X = Z
    else:
        X = np.concatenate([Z, X], axis=1)

    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.3, random_state=3)
    o, o_ = predict_outcome(y_train, y_test, X_train, X_test)
    roc = {'type': option, 'train': o, 'test': o_}
    return roc


roc_table = pd.DataFrame(columns=['type', 'train', 'test'])
roc_table = roc_table.append(predictive_models_comp(1, ks, data_df_, variants_f, clinical), ignore_index=True)
roc_table = roc_table.append(predictive_models_comp(2, ks, data_df_, variants_f, clinical), ignore_index=True)
roc_table = roc_table.append(predictive_models_comp(3, ks, data_df_, variants_f, clinical), ignore_index=True)
roc_table = roc_table.append(predictive_models_comp(4, ks, data_df_, variants_f, clinical), ignore_index=True)
roc_table = roc_table.append(predictive_models_comp(5, ks, data_df_, variants_f, clinical), ignore_index=True)

key = ['All SNPS', 'Known SNPS', 'Confirmed SNPS', 'Clinical', 'Random']
for i in range(roc_table.shape[0]):
    print('\n\nModel', key[i], ':\nTest:', roc_table.test[i], '\nTrain:', roc_table.train[i])






"""
Written by Raquel Aoki and Gabriel Oliveira
Date: December 2020
"""
import sys
sys.path.append("/src")
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from torchmeta.modules import MetaModule
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
import copy
import yaml
import time
import warnings
from tqdm import tqdm

#local libraries
from mmoe import MMoE, MMoETowers, MMoEEx
from standard_optimization import standarOptimization
from task_balancing import TaskBalanceMTL
from utils import *
from data_preprocessing import *

warnings.filterwarnings("ignore")

def main(config_path):
    """Start: Parameters Loading"""
    with open(config_path) as f:
        config = yaml.load_all(f, Loader=yaml.FullLoader)
        for p in config:
            params = p["parameters"]

    try:
        SEED = params["SEED"]
    except KeyError:
        SEED = 2

    # Fix numpy seed for reproducibility
    np.random.seed(SEED)
    # Fix random seed for reproducibility
    random.seed(SEED)
    # Fix Torch graph-level seed for reproducibility
    torch.manual_seed(SEED)

    if params["runits"][0] == "None":
        params["runits"] = None

    try:
        print(params["rep_ci"])
    except KeyError:
        params["rep_ci"] = 1

    if params["rep_ci"] > 1:
        ci_test = []
    """End: Parameters Loading"""


    """Start: Tensorboard creation"""
    if params["save_tensor"]:
        path_logger = "mtl-research-data-fall2020/tensorboard/"
    else:
        path_logger = "mtl-research-data-fall2020/notsave/"
    config_name = (
        "model_"
        + params["model"]
        + "_Ntasks_"
        + str(len(params["tasks"]))
        + "_batch_"
        + str(params["batch_size"])
        + "_N_experts_"
        + str(params["num_experts"])
    )
    if params["save_tensor"]:
        writer_tensorboard = TensorboardWriter(path_logger, config_name)

    # starting date to folder creation
    date_start = datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
    """End: Tensorboard Creation"""


    for rep in range(params["rep_ci"]):
        #print("\nRepetition {}".format(rep))
        rep_start = time.time()
        """Start: Data Loading"""
        print('Starting Data Preparation...')
        if params["data"] == "census":
            print("... Census Data")
            (
                train_loader,
                validation_loader,
                test_loader,
                num_features,
                num_tasks,
                output_info,
            ) = data_preparation_census(params)
            X, y = next(iter(train_loader))
            params["seqlen"] = None
            params['expert'] = None
        elif params["data"] == "mimic":
            print(".... MIMIC data")
            (
                train_loader,
                validation_loader,
                test_loader,
                task_info,
                task_number,
            ) = data_preparation_mimic3(params["batch_size"], params["seqlen"], params["prop"])
            X, pheno, los, decomp, ihm = next(iter(train_loader))
        elif params['data'] == 'pcba':
            print('... PCBA dataset')
            try:
                print('seqlen',params["seqlen"])
                params["seqlen"] = None
            except KeyError:
                params["seqlen"] = None

            (
                train_loader,
                validation_loader,
                test_loader,
                num_features,
                num_tasks,
                params['tasks']
            ) = data_preparation_pcba(params)
            X, y, w = next(iter(train_loader))
            print('...batch shapes: ', X.shape,y.shape,w.shape)
            #print('Are tasks balance?',X.shape,y.sum(0))
            params["lambda"] = np.repeat(1, num_tasks)
        else:
            """
            Check data_preprocessing.py for examples of load functions
            """
            print('... your new dataset')
            (
                train_loader,
                validation_loader,
                test_loader,
                num_features,
                num_tasks,
                params['tasks'],
                #buckets
            ) = data_preparation_newdataset(params)
            X, y = next(iter(train_loader))
            print('...batch shapes: ', X.shape,y.shape)

        print('Done Data Preparation!')
        """End: Data Loading"""

        """Start: Model Initialization and Losses"""
        if params["data"] == "census" or params['data'] == 'pcba':
            #Option for non-temporal data
            if params["model"] == "Standard":
                #Shared bottom
                model = standarOptimization(
                    data=params['data'],
                    num_units=params["num_units"],
                    num_experts=params["num_experts"],
                    num_tasks=len(params["tasks"]),
                    num_features=num_features,
                    hidden_size=params["hidden_size"],
                    tasks_name=params["tasks"],
                    runits=params["runits"],
                )
            else:
                #MMoEEx or MMoE
                model = MMoETowers(
                    data=params['data'],
                    tasks_name=params["tasks"],
                    num_tasks=num_tasks,
                    num_experts=params["num_experts"],
                    num_units=params["num_units"],
                    num_features=num_features,
                    modelname=params["model"],
                    runits=params["runits"],
                    prob_exclusivity=params["prob_exclusivity"],
                    type=params["type_exc"],
                    expert=params['expert'],
                )

            if params['data']=='census':
                try:
                    print('... class weights: ',params["cw_census"])
                except KeyError:
                    params["cw_census"] = np.repeat(1, num_tasks)
                    print('... class weights: ',params["cw_census"])

                criterion = [
                    nn.BCEWithLogitsLoss(pos_weight=torch.tensor(params["cw_census"][i]).to(device)) for i in range(num_tasks)
                ]
                lr = 1e-4 #Learning Rate
                wd = 0.01 # Adam weight_decay

            else:
                criterion = [
                    nn.BCEWithLogitsLoss(pos_weight=torch.tensor(params["cw_pcba"]).to(device)) for i in range(num_tasks)
                ]
                lr = 0.003 #Learning Rate
                wd = 0.001 # Adam weight_decay
        elif params["data"] == "mimic":
            #Option for temporal data
            params["expert_blocks"] = try_keyerror('expert_blocks',params)
            if params["model"] == "Standard":
                #Shared Bottom
                model = standarOptimization(
                    data=params['data'],
                    num_units=params["num_units"],
                    num_experts=params["num_experts"],
                    num_tasks=len(params["tasks"]),
                    num_features=X.shape[2],
                    hidden_size=params["hidden_size"],
                    learning_type="LSTM",
                    seqlen=X.shape[1],
                    task_number=task_number,
                    expert=params["expert"],
                    task_info=task_info,
                    tasks_name=params["tasks"],
                    runits=params["runits"],
                )
            else:
                #MMoEEx or MMoE
                model = MMoETowers(
                    data=params['data'],
                    tasks_name=params["tasks"],
                    num_tasks=len(params["tasks"]),
                    num_experts=params["num_experts"],
                    num_units=params["num_units"],
                    num_features=X.shape[2],
                    modelname=params["model"],
                    task_info=task_info,
                    task_number=task_number,
                    runits=params["runits"],
                    expert=params["expert"],
                    expert_blocks=params["expert_blocks"],
                    seqlen=X.shape[1],
                    n_layers=params["lstm_nlayers"],
                    prob_exclusivity=params["prob_exclusivity"],
                    type=params["type_exc"],
                )
            wd = 0.0001  # Adam weight_decay
            lr = 1e-3 #Learning Rate
            criterion = []
            for task in params["tasks"]:
                if task == "los":
                    # investigating if these help, results are not as good as the paper, I'm not sure why
                    criterion.append(
                        nn.CrossEntropyLoss(weight=torch.tensor(params["cw_los"]).to(device).float())
                    )  #
                elif task == "pheno":
                    criterion.append(
                        nn.BCEWithLogitsLoss(
                            pos_weight=torch.tensor(params["cw_pheno"]).to(device)
                        )
                    )
                elif task == "ihm":
                    criterion.append(
                        nn.BCEWithLogitsLoss(
                            pos_weight=torch.tensor(params["cw_ihm"]).to(device)
                        )
                    )
                else:
                    # decomp is very unbalanced
                    criterion.append(
                        nn.BCEWithLogitsLoss(
                            pos_weight=torch.tensor(params["cw_decomp"]).to(device)
                        )
                    )
        else:
            #your new dataset
            params["expert_blocks"] = try_keyerror('expert_blocks',params)
            if params["model"] == "Standard":
                #Shared Bottom
                model = standarOptimization(
                    #add your parameters
                )
            else:
                #MMoEEx or MMoE
                model = MMoETowers(
                    #add your parameters
                )
            wd = 0.0001  # Adam weight_decay
            lr = 1e-3 #Learning Rate
            #change to your criterion
            criterion = [
                    nn.BCEWithLogitsLoss().to(device) for i in range(num_tasks)
                ]

        if torch.cuda.is_available():
            model.to(device)
        """end: Model Initialization"""

        """Start: Variables Initialization"""
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=wd
        )
        if params['data']!= 'pcba':
            opt_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=params["gamma"]
            )

        #OPTIONAL: DWA and LBTW task balancing parameters and initialization
        if params['data'] =='pcba':
            try:
                alpha = params['dwa_alpha']
            except KeyError:
                alpha = 0.5
        else:
            alpha = 0.5
        balance_tasks = TaskBalanceMTL(model.num_tasks, params["task_balance_method"],alpha_balance = alpha)

        #loss variables initialization
        loss_ = []
        task_losses = np.zeros([model.num_tasks], dtype=np.float32)
        best_val_AUC = 0
        best_epoch = 0
        """End: Variables Initialization"""


        print("START TRAINING")
        for e in range(params["max_epochs"]):
            torch.cuda.empty_cache()
            for i, batch in enumerate(tqdm(train_loader)):
                optimizer.zero_grad()
                loss = 0

                if params["model"] == "Standard":
                    train_y_pred = model(batch[0].to(device))
                    for task in range(model.num_tasks):
                        if params["data"] == "census":
                            label = (
                                batch[1][:, task].long().to(device).reshape(-1, 1)
                            )
                            task_losses[task] = criterion[task](train_y_pred[task], label.float())*params["lambda"][task]
                            loss += criterion[task](train_y_pred[task], label.float())*params["lambda"][task]
                        elif params['data'] == 'pcba':
                            label = (
                                batch[1][:, task].long().to(device).reshape(-1, 1)
                            )
                            loss_temp = criterion[task](train_y_pred[task][batch[2][:,task]>0], label.float()[batch[2][:,task]>0])*params["lambda"][task]
                            loss += loss_temp
                            if params["task_balance_method"] is not None and loss_temp < 0.0001:
                                task_losses[task] = 0.0001
                            else:
                                task_losses[task] = loss_temp
                        elif params['data'] == 'mimic':
                            #time series
                            (train_y_pred, label_inner, _, _) = maml_split(
                                batch, model, device, prop=1, time=True
                            )
                            pred, obs = organizing_predictions(
                                model, params, train_y_pred[task], label_inner, task
                            )

                            task_losses[task] = criterion[task](pred, obs).float()*params["lambda"][task]
                            loss += criterion[task](pred, obs).float()*params["lambda"][task]
                        else:
                            #your new model/dataset
                            #save the current task's labels
                            label = (
                                batch[1][:, task].long().to(device).reshape(-1, 1)
                            )
                            #calculate loss using the criterion function
                            loss_temp = criterion[task](train_y_pred[task][batch[2][:,task]>0], label.float()[batch[2][:,task]>0])*params["lambda"][task]
                            loss += loss_temp
                    loss.backward()
                    optimizer.step()
                else:
                    '''
                    MMoE or MMoEEx-diversity
                    - MMoE: Multi-gate Mixture-of-Experts
                    - Md: Multi-gate Mixture-of-Experts with Exclusivity (only diversity component, without MAML-MTl optimization)
                    '''
                    '''
                    MMoEEx: Diversity + MAML-MTL (Full model)
                    '''
                    if params["model"] == "MMoE" or params['model']== 'Md':
                        train_y_pred = model(batch[0].to(device))
                        for task in range(model.num_tasks):
                            if params["data"] == "census":
                                label = (
                                    batch[1][:, task].long().to(device).reshape(-1, 1)
                                )
                                loss += criterion[task](
                                    train_y_pred[task], label.float()
                                )*params["lambda"][task]
                                #saving loss per task for task-balancing
                                task_losses[task] = criterion[task](
                                    train_y_pred[task], label.float()
                                )*params["lambda"][task]
                            elif params['data'] == 'pcba':
                                label = (
                                    batch[1][:, task].long().to(device).reshape(-1, 1)
                                )
                                loss_temp = criterion[task](
                                    train_y_pred[task][batch[2][:,task]>0], label.float()[batch[2][:,task]>0]
                                )*params["lambda"][task]
                                loss += loss_temp
                                #saving loss per task for task-balancing
                                task_losses[task] = loss_temp
                            elif params['data'] == 'mimic':
                                # this maml split here is to reorganize the data nicely
                                (train_y_pred, label_inner, _, _) = maml_split(
                                    batch, model, device, prop=1, time=True
                                )
                                pred, obs = organizing_predictions(
                                    model, params, train_y_pred[task], label_inner, task
                                )
                                loss += criterion[task](pred, obs).float()*params["lambda"][task]
                                #saving loss per task for task-balancing
                                task_losses[task] = criterion[task](pred, obs).float()*params["lambda"][task]
                            else:
                                #your new model/dataset
                                label = (
                                    batch[1][:, task].long().to(device).reshape(-1, 1)
                                )  # .cuda()
                                loss += criterion[task](
                                    train_y_pred[task], label.float()
                                )
                                #saving loss per task for task-balancing
                                task_losses[task] = criterion[task](
                                    train_y_pred[task], label.float()
                                )*params["lambda"][task]
                        loss.backward()
                        if params['model']=='Md':
                            #Keeping gates 'closed'
                            if params["type_exc"] == "exclusivity":
                                (
                                    model.MMoEEx.gate_kernels.grad.data,
                                    model.MMoEEx.gate_bias.grad.data,
                                ) = keep_exclusivity(model)
                            else:
                                #Exclusion
                                (
                                    model.MMoEEx.gate_kernels.grad.data,
                                    model.MMoEEx.gate_bias.grad.data,
                                ) = keep_exclusion(model)
                        optimizer.step()

                    else:
                        '''
                        1) For MAML-MTL, we split the training set in inner and outer loss calculation
                        '''
                        if params["data"] == "census":
                            (
                                train_y_pred,
                                label_inner,
                                train_outer,
                                label_outer,
                            ) = maml_split(
                                batch, model, device, params["maml_split_prop"]
                            )
                        elif params['data'] == 'pcba':
                            (
                                train_y_pred,
                                label_inner,
                                weight_inner,
                                train_outer,
                                label_outer,
                                weight_outer
                            ) = maml_split(
                                batch, model, device, params["maml_split_prop"], data_pcba = True
                            )
                        elif params['data'] == 'mimic':
                            (
                                train_y_pred,
                                label_inner,
                                train_outer,
                                label_outer,
                            ) = maml_split(
                                batch,
                                model,
                                device,
                                params["maml_split_prop"],
                                True,
                                params["seqlen"],
                            )
                        else:
                            #your new model/dataset
                            (
                                train_y_pred,
                                label_inner,
                                train_outer,
                                label_outer,
                            ) = maml_split(
                                batch, model, device, params["maml_split_prop"]
                            )

                        loss_task_train = []
                        '''
                        2) Deepcopy to save the model before temporary updates
                        '''
                        model_copy = copy.deepcopy(model)
                        for task in range(model.num_tasks):
                            '''
                            3) Inner loss / loss in the current model
                            '''
                            pred, obs = organizing_predictions(
                                model, params, train_y_pred[task], label_inner, task
                            )
                            inner_loss = criterion[task](pred, obs).float()
                            '''
                            4) Temporary update per task
                            '''
                            params_ = gradient_update_parameters(model, inner_loss, step_size = optimizer.param_groups[0]['lr'])
                            '''
                            5) Calculate outer loss / loss in the temporary model
                            '''
                            current_y_pred = model(train_outer, params=params_)
                            pred, obs = organizing_predictions(
                                model, params, current_y_pred[task], label_outer, task,
                            )
                            loss_out = (
                                criterion[task](pred, obs).float()
                                * params["lambda"][task]
                            )
                            task_losses[task] = loss_out
                            loss += loss_out
                            loss_task_train.append(loss_out.cpu().detach().numpy())
                            '''
                            6) Reset temporary model
                            '''
                            for (_0, p_), (_1, p_b) in zip(model.named_parameters(), model_copy.named_parameters()):
                                p_.data = p_b.data

                        loss.backward()
                        #Keeping gates 'closed'
                        if params["type_exc"] == "exclusivity":
                            (
                                model.MMoEEx.gate_kernels.grad.data,
                                model.MMoEEx.gate_bias.grad.data,
                            ) = keep_exclusivity(model)
                        else:
                            (
                                model.MMoEEx.gate_kernels.grad.data,
                                model.MMoEEx.gate_bias.grad.data,
                            ) = keep_exclusion(model)
                        optimizer.step()

                """ Optional task balancing step"""
                if params["task_balance_method"] == "LBTW":
                    for task in range(model.num_tasks):
                        if i == 0:  # first batch
                            balance_tasks.get_initial_loss(
                                task_losses[task],
                                task,
                            )
                        balance_tasks.LBTW(task_losses[task], task)
                        weights = balance_tasks.get_weights()
                        params["lambda"] = weights

            if params["task_balance_method"] == "LBTW":
                print('... Current weights LBTW: ',params["lambda"])

            """ Saving losses per epoch"""
            loss_.append(loss.cpu().detach().numpy())

            print("... calculating metrics")
            if params["data"] == "census":
                print('Validation')
                auc_val, _, loss_val = metrics_census(e, validation_loader, model, device, criterion)
                print('Train')
                auc_train, _ = metrics_census(e, train_loader, model, device, train = True)
            elif params['data']=='pcba':
                print('Validation')
                auc_val, _, loss_val = metrics_pcba(e, validation_loader, model, device, criterion)
                print('Train')
                auc_train, _ = metrics_pcba(e, train_loader, model, device, train=True)
            elif params["data"] == "mimic":
                auc_val, loss_val, _ = metrics_mimic(e,validation_loader,model,device,params["tasks"],criterion,validation=True)
                auc_train, _, _ = metrics_mimic(e, train_loader, model, device, params["tasks"], [], training=True)
            else:
                print('Validation')
                auc_val, _, loss_val = metrics_newdata(e, validation_loader, model, device, criterion)
                print('Train')
                auc_train, _ = metrics_newdata(e, train_loader, model, device, train = True)


            """Updating tensorboard"""
            if params["save_tensor"]:
                writer_tensorboard.add_scalar(
                    "Loss/train_Total", loss.cpu().detach().numpy(), e
                )
                for task in range(model.num_tasks):
                    writer_tensorboard.add_scalar(
                        "Auc/train_T" + str(task + 1), auc_train[task], e
                    )
                    writer_tensorboard.add_scalar(
                        "Auc/Val_T" + str(task + 1), auc_val[task], e
                    )
                writer_tensorboard.end_writer()

            """Printing Outputs """
            if e % 1 == 0:
                if params["gamma"] < 1 and e % 10 == 0 and e>1:
                    opt_scheduler.step()
                if params["rep_ci"] <= 1 and params['data'] != 'pcba':
                    print(
                        "\n{}-Loss: {} \nAUC-Train: {}  \nAUC-Val: {} \nL-val: {}".format(
                            e, loss.cpu().detach().numpy(), auc_train, auc_val, loss_val
                        )
                    )
                elif params["rep_ci"] <= 1 and params['data'] == 'pcba':
                    print(
                        "\n{}-Loss: {} \nAUC-Train-pcba: {}  \nAUC-Val: {} \nL-val: {}".format(
                            e, loss.cpu().detach().numpy(), np.nanmean(auc_train), np.nanmean(auc_val), np.mean(loss_val)
                        )
                    )

            """Saving the model with best validation AUC"""
            if params["best_validation_test"]:
                current_val_AUC = np.nansum(auc_val)
                if current_val_AUC > best_val_AUC:
                    best_epoch = e
                    best_val_AUC = current_val_AUC
                    print("better AUC ... saving model")
                    # path to save model
                    path = (
                        ".//output//"
                        + date_start
                        + "/"
                        + params["model"]
                        + "-"
                        + params["data"]
                        + "-"
                        + str(params["num_experts"])
                        + "-"
                        + params["output"]
                        + "/"
                    )

                    Path(path).mkdir(parents=True, exist_ok=True)
                    path = path + "net_best.pth"
                    torch.save(model.state_dict(), path)
                print('...best epoch',best_epoch)

            """Optional: DWA task balancing"""
            if params["task_balance_method"] == "DWA":
                # add losses to history structure
                balance_tasks.add_loss_history(task_losses)
                balance_tasks.last_elements_history()
                balance_tasks.DWA(task_losses, e)
                weights = balance_tasks.get_weights()
                params["lambda"] = weights
                print('... Current weights DWA: ',params["lambda"])

            """Reset array with loss per task"""
            task_losses[:] = 0.0


        loss_ = np.array(loss_).flatten().tolist()
        torch.cuda.empty_cache()

        if params["best_validation_test"]:
            print("...Loading best validation epoch")
            path = (
                ".//output//"
                + date_start
                + "/"
                + params["model"]
                + "-"
                + params["data"]
                + "-"
                + str(params["num_experts"])
                + "-"
                + params["output"]
                + "/"
            )
            path = path + "net_best.pth"
            model.load_state_dict(torch.load(path))

        print('... calculating metrics on testing set')
        if params["data"] == "census":
            auc_test, conf_interval = metrics_census(epoch=e,data_loader=test_loader,model=model,device=device,confidence_interval=True)
        elif params['data'] == 'pcba':
            auc_test, conf_interval = metrics_pcba(epoch=e,data_loader=test_loader,model=model,device=device,confidence_interval=True)
            precision_auc_test = np.repeat(0,model.num_tasks)
        elif params["data"] == "mimic":
            auc_test, _, conf_interval = metrics_mimic(epoch=e,data_loader=test_loader,model=model,device=device,tasksname=params["tasks"],criterion=[],testing=True)
        else:
            auc_test, _, conf_interval = metrics_newdata(epoch=e,data_loader=test_loader,model=model,device=device,confidence_interval=True)

        print('... calculating diversity on testing set experts')
        measuring_diversity(test_loader, model, device,params['output'],params['data'])

        """Creating and saving output files"""
        if params["rep_ci"] <= 1:
            if params['data']=='pcba':
                print("\nFinal AUC-Test: {}".format(np.nanmean(auc_test)))
            else:
                print("\nFinal AUC-Test: {}".format(auc_test))

            print("...Creating the output file")
            if params["create_outputfile"]:
                if params['data'] != 'pcba':
                    precision_auc_test = ''

                data_output = output_file_creation(
                    rep,
                    model.num_tasks,
                    auc_test,
                    auc_val,
                    auc_train,
                    conf_interval,
                    rep_start,
                    params,
                    precision_auc_test,
                )
                path = (
                    ".//output//"
                    + date_start
                    + "/"
                    + params["model"]
                    + "-"
                    + params["data"]
                    + "-"
                    + str(params["num_experts"])
                    + "-"
                    + params["task_balance_method"]
                    + "/"
                )
                data_output.to_csv(path+params['output']+'.csv', header = True,index=False)

        else:
            ci_test.append(auc_test)
            if params["create_outputfile"]:
                if rep == 0:
                    data_output = output_file_creation(
                        rep,
                        model.num_tasks,
                        auc_test,
                        auc_val,
                        auc_train,
                        conf_interval,
                        rep_start,
                        params,
                    )
                    path = (
                        ".//output//"
                        + date_start
                        + "/"
                        + params["model"]
                        + "-"
                        + params["data"]
                        + "-"
                        + str(params["num_experts"])
                        + "-"
                        + params["task_balance_method"]
                        + "/"
                    )

                    data_output.to_csv(path+params['output']+'.csv', header = True,index=False)
                else:
                    _output = {"repetition": rep}
                    for i in range(model.num_tasks):
                        colname = "Task_" + str(i)
                        _output[colname + "_test"] = auc_test[i]
                        _output[colname + "_test_bs_l"] = conf_interval[i][0]
                        _output[colname + "_test_bs_u"] = conf_interval[i][1]
                        _output[colname + "_val"] = auc_val[i]
                        _output[colname + "_train"] = auc_train[i]
                    _output["time"] = time.time() - rep_start
                    _output["params"] = params
                    _output["data"] = params["data"]
                    _output["tasks"] = params["tasks"]
                    _output["model"] = params["model"]
                    _output["batch_size"] = params["batch_size"]
                    _output["max_epochs"] = params["max_epochs"]
                    _output["num_experts"] = params["num_experts"]
                    _output["num_units"] = params["num_units"]

                    _output["expert"] = try_keyerror("expert", params)
                    _output["expert_blocks"] = try_keyerror("expert_blocks", params)
                    _output["seqlen"] = try_keyerror("seqlen", params)

                    #_output["use_early_stop"] = params["use_early_stop"]
                    _output["runits"] = params["runits"]

                    _output["prop"] = params["prop"]
                    _output["lambda"] = params["lambda"]
                    _output["cw_pheno"] = try_keyerror("cw_pheno", params)
                    _output["cw_decomp"] = try_keyerror("cw_decomp", params)
                    _output["cw_ihm"] = try_keyerror("cw_ihm", params)
                    _output["cw_los"] = try_keyerror("cw_los", params)
                    _output["cw_pcba"] = try_keyerror("cw_pcba", params)
                    _output["lstm_nlayers"] = try_keyerror("lstm_nlayers", params)
                    _output["task_balance_method"] = params["task_balance_method"]
                    _output["type_exc"] = params["type_exc"]
                    _output["prob_exclusivity"] = params["prob_exclusivity"]
                    #_output["clustering"] = try_keyerror("clustering", params)
                    #_output["buckets"] = try_keyerror("buckets", params)


                    data_output = data_output.append(_output, ignore_index=True)
                    '''
                    path = (
                        ".//output//"
                        + date_start
                        + "/"
                        + params["model"]
                        + "-"
                        + params["data"]
                        + "-"
                        + str(params["num_experts"])
                        + "-"
                        + params["task_balance_method"]
                        + "/"
                    )
                    data_output.to_csv(
                        path
                        + "/"
                        + datetime.today().strftime("%Y-%m-%d-%H:%M:%S")
                        + ".csv",
                        header=True,
                        index=False,
                    )'''
                    data_output.to_csv('fall2020//output//'+params['output']+'.csv', header = True,index=False)

    """Calculating the Confidence Interval using Bootstrap"""
    if params["rep_ci"] > 1:
        model_CI(ci_test, model)
    print('...Best Epoch: ',best_epoch)
    print(params)


if __name__ == "__main__":
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Cuda Availble:", torch.cuda.is_available(), " device: ", device)
    if len(sys.argv)==1:
        print('Testing:')
        passed = 0
        for i in range(7):
            try:
                main(config_path='testing/config_test'+str(i)+'.yaml')
                print("========================================================================")
                print("========================================================================")
                print("========================================================================")
                print("PASS TEST ",str(i),'!!!')
                print("========================================================================")
                print("========================================================================")
                print("========================================================================")
                passed+=1
            except:
                print("========================================================================")
                print("========================================================================")
                print("========================================================================")
                print("FAILED TEST ",str(i))
                print("========================================================================")
                print("========================================================================")
                print("========================================================================")

        print("========================================================================")
        print("========================================================================")
        print("========================================================================")
        print('TESTING RESULTS: Code pass in ',str(round(passed*100/7),2),' of the tests.')
        print("========================================================================")
        print("========================================================================")
        print("========================================================================")

    else:
        main(config_path=sys.argv[1])
    end_time = time.time() - start_time
    end_time_m = end_time / 60
    end_time_h = end_time_m / 60
    print("Time ------ {} min / {} hours ------".format(end_time_m, end_time_h))
