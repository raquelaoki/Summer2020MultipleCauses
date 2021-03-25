"""
Author: Raquel Aoki, March 2021
Update from colab and main_old.py
"""
from typing import Any

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sys
import train as parkca
import eval as evaluation

data_preprocessing = False
run_learners = True


def main(config_path, DataPreprocessing=False, RunLearners=True):
    path = '/content/'
    with open(config_path) as f:
        config = yaml.load_all(f, Loader=yaml.FullLoader)
        for p in config:
            params = p["parameters"]

    """ Data pre-processing or loading data already pre-processed """
    # 318 (318, 42) (318, 16641) 16641
    if DataPreprocessing:
        print('Option 1: Starting pre-processing:')
        import data_preprossing_functions as dpf
        y, x_clinical, x_snps, x_colnames, x_clinical_names = dpf.run_preprocessing(path, True)
        x_clinical = x_clinical.astype('float64')
        print('Done!')
        print('Shapes:', len(y), sum(y), x_clinical.shape, x_snps.shape, len(x_colnames), len(x_clinical_names))
    else:
        print('Option 2: Reading files')
        y = np.load('y.npy')
        x_clinical = np.load('x_clinical.npy', allow_pickle=True)
        x_clinical = x_clinical.astype('float64')
        x_clinical_names = np.load('x_clinical_names.npy', allow_pickle=True)
        x_snps = np.load('x_snps.npy', allow_pickle=True)
        x_colnames = np.load('x_colnames.npy', allow_pickle=True)
        print('Done!')
        print('Shapes:', len(y), sum(y), x_clinical.shape, x_snps.shape, len(x_colnames), len(x_clinical_names))

    """ Parkca: Learners """
    if RunLearners:
        print("Running Learners!")
        if 'BART' in params['learners']:
            try:
                sys.path.insert(0, 'bartpy/')
                from bartpy.sklearnmodel import SklearnModel
            except NameError:
                print('BART Library Missing')
                print("Check: https://github.com/JakeColtman/bartpy")
                sys.exit()
        X = np.concatenate([x_clinical, x_snps], axis=1)
        print(X.shape)

        X = preprocessing.StandardScaler().fit_transform(X)
        level1data = parkca.learners([params['learners']], X, y, x_colnames)
        print(level1data.head())
        print('Learners: Done!')

    else:
        print('Loading option not implemented for this option')
        sys.exit()

    """ Parkca: Meta-learners"""
    # TODO: add meta-learner
    ks = pd.read_csv(path + 'known_snps_fullname.txt', header=None)[0].values
    level1data['y'] = 0
    level1data['y'] = [1 if i in ks else 0 for i in level1data['causes'].values]

    # Checking the Diversity
    diversity_mean, _ = evaluation.diversity(params['learners'], level1data, 'y')
    print('Diversity (Best when <= 0): ', diversity_mean)

    level1data.set_index('causes', inplace=True, drop=True)
    level1data = parkca.data_norm(level1data)
    roc, MetalearnerOutput, y_full_prob = parkca.meta_learner(level1data, params['metalearners'], 'y')

    # Random Forest (RF) had the best results on tests
    # Creating Model's predictions
    output_potential_causes, output_potential_causes_signal = [], []
    output_missing_causes, output_missing_causes_signal, output_confirmed_causes = [], [], []
    for i in range(len(MetalearnerOutput['rf'])):
        if MetalearnerOutput['rf'][i] == 1:
            output_potential_causes.append(MetalearnerOutput.index[i])
            output_potential_causes_signal.append(y_full_prob[i, 1])
            if MetalearnerOutput.index[i] in ks:
                output_confirmed_causes.append(MetalearnerOutput.index[i])
        elif MetalearnerOutput['rf'][i] == 0 and MetalearnerOutput.index[i] in ks:
            output_missing_causes.append(MetalearnerOutput.index[i])
            output_missing_causes_signal.append(y_full_prob[i, 1])
        else:
            pass
    print("Missing Causes:", len(output_missing_causes))
    print("Confirmed Causes:", len(output_confirmed_causes))
    print("New Potential Causes:", len(output_potential_causes)-len(output_confirmed_causes))

    # TODO: add clinical analsys


if __name__ == '__main__':
    # main(config_path = sys.argv[1])
    main(config_path='/content/')
