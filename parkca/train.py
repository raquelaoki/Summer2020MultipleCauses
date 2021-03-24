"""
MODIFICATION FOR BC CHILDREN HOSPITAL
"""
import pandas as pd
import numpy as np
import warnings
import os
import eval as eval
from os import listdir
from os.path import isfile, join
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn import linear_model
from sklearn import calibration
from scipy import sparse, stats
import functools
from torch.utils.data import Dataset, DataLoader
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
import tensorflow.compat.v1 as tf
from tensorflow.keras import optimizers
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd  # conda install -c conda-forge tensorflow-probability
from bartpy.sklearnmodel import SklearnModel

tf.disable_v2_behavior()
tf.enable_eager_execution()
warnings.simplefilter("ignore")

# import datapreprocessing as dp
# import CEVAE as cevae
# DA
# Meta-leaners packages
# https://github.com/aldro61/pu-learning (clone)
# from puLearning.puAdapter import PUAdapter
# https://github.com/t-sakai-kure/pywsl
# from pywsl.pul import pu_mr #pumil_mr (pip install pywsl)
# from pywsl.utils.syndata import gen_twonorm_pumil
# from pywsl.utils.comcalc import bin_clf_err

def learners(LearnersList, X, y, colnamesX, id='', Z=None, colnamesZ=None, path_output=None):
    """
    input:
        path_output: where to save the files
        X, colnamesX: potential causes and their names
        Z, colnamesZ: confounders and their names (clinical)
        y: 01 outcome
        causes: name of the potential causes (snps)
    """
    roc_table = pd.DataFrame(columns=['learners', 'fpr', 'tpr', 'auc'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    coef_table = pd.DataFrame(columns=['causes'])
    coef_table['causes'] = colnamesX

    if 'DA' in LearnersList:
        print('\n\nLearner: DA')
        k_list = [15]  # if exploring multiple latent sizes
        print('... There are ', len(k_list), ' versions of DA')
        b = 100
        for k in k_list:
            print('Version 1/', len(k_list))
            coln = 'DA_'+str(id)+str(k)
            #coefk_table = pd.DataFrame(columns=[causes])
            model_da = learner_deconfounder_algorithm(X_train, X_test, y_train, y_test, 10)
            coef, coef_continuos, roc = model_da.fit()
            roc_table = roc_table.append(roc, ignore_index=True)

            coef_table[coln] = coef_continuos[0:len(colnamesX)]
            print('Done!')
    if 'BART' in LearnersList:
        print('\n\nLearner: BART')
        # model = SklearnModel(n_trees=50, n_burn=50, n_chains=1, n_jobs=1)  # Use default parameters
        # model.fit(x_snps, y)  # Fit the model
        model_bart = learner_BART(X_train, X_test, y_train, y_test)
        model_bart.fit()
        print('...... predictions')
        coef_table['BART'] = ''
        # predictions = model.predict(x_snps)  # [:,0:1000] Make predictions on the train set
        # print(predictions[0])
        #TODO: CATE
    return coef_table

def classification_models(y, y_, X, X_, name_model, y_full, X_full):
    """
    Meta-learner
    Input:
        X,y,X_test, y_test: dataset to train the model
    Return:
        cm: confusion matrix for the testing set
        cm_: confusion matrix for the full dataset
        y_all_: prediction for the full dataset
    """
    # X_full = np.concatenate((X,X_), axis = 0 )
    # y_full = np.concatenate((y,y_), axis = 0)

    warnings.filterwarnings("ignore")
    if name_model == 'nn':
        y_pred, ypred = nn_classifier(y, X, X_, X_full.values)
        pr = precision(1, confusion_matrix(y_, y_pred))
        f1 = f1_score(y_, y_pred)
        if np.isnan(pr) or pr == 0 or f1 < 0.01:
            while np.isnan(pr) or pr == 0 or f1 < 0.001:
                print('\n\n trying again \n\n')
                y_pred, ypred = nn_classifier(y, X, X_, X_full)
                pr = precision(1, confusion_matrix(y_, y_pred))
                f1 = f1_score(y_, y_pred)
                print(f1, confusion_matrix(y_, y_pred))
        print('\n Model:', name_model, ':\nFull set: tn, fp, fn, tp\n', confusion_matrix(y_full, ypred))
        print('\nPrecision (test)', precision(1, confusion_matrix(y_, y_pred)))
        print('Recall (test)', recall(1, confusion_matrix(y_, y_pred)))

    else:

        if name_model == 'adapter':
            estimator = SVC(C=100, kernel='rbf', gamma='scale', probability=True)  # C = 0.3
            model = PUAdapter(estimator, hold_out_ratio=0.1)
            X = np.matrix(X)
            y0 = np.array(y)
            y0[np.where(y0 == 0)[0]] = -1
            model.fit(X, y0)

        elif name_model == 'upu':
            '''
            pul: nnpu (Non-negative PU Learning), pu_skc(PU Set Kernel Classifier),
            pnu_mr:PNU classification and PNU-AUC optimization (the one tht works: also use negative data)
            nnpu is more complicated (neural nets, other methos seems to be easier)
            try https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/pu_skc/demo_pu_skc.py
            and https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/upu/demo_upu.py
             '''
            print('upu', X.shape[1])
            prior = .48  # change for the proportion of 1 and 0
            param_grid = {'prior': [prior],
                          'lam': np.logspace(-3, 3, 5),  # what are these values
                          'basis': ['lm']}
            # upu (Unbiased PU learning)
            # https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/upu/demo_upu.py
            model = GridSearchCV(estimator=pu_mr.PU_SL(),
                                 param_grid=param_grid, cv=3, n_jobs=-1)
            X = np.matrix(X)
            y = np.array(y)
            model.fit(X, y)

        elif name_model == 'lr':
            print('lr', X.shape[1])
            X = np.matrix(X)
            y = np.array(y)
            from sklearn.linear_model import LogisticRegression
            w1 = y.sum() / len(y)
            w0 = 1 - w1
            sample_weight = {0: w1, 1: w0}
            model = LogisticRegression(C=0.1, penalty='l2', class_weight=sample_weight)  #
            model.fit(X, y)

        elif name_model == 'rf':
            print('rd', X.shape[1])
            w = len(y) / y.sum()
            sample_weight = np.array([w if i == 1 else 1 for i in y])
            model = RandomForestClassifier(max_depth=6, random_state=0)
            model.fit(X, y, sample_weight=sample_weight)

        else:
            print('random', X.shape[1])

        if name_model == 'random':
            p = y.sum() + y_.sum()
            p_full = p / (len(y) + len(y_))
            y_pred = np.random.binomial(n=1, p=y_.sum() / len(y_), size=X_.shape[0])
            ypred = np.random.binomial(n=1, p=p_full, size=X_full.shape[0])
        else:
            y_pred = model.predict(X_)
            ypred = model.predict(X_full)
            y_prob_full = model.predict_proba(X_full)

            y_pred = np.where(y_pred == -1, 0, y_pred)
            ypred = np.where(ypred == -1, 0, ypred)

            print('\n Model:', name_model, ':\nFull set: tn, fp, fn, tp\n', confusion_matrix(y_full, ypred).ravel())
            print('\nTest  set: tn, fp, fn, tp\n', confusion_matrix(y_, y_pred).ravel())
            print('\nPrecision (test)', precision(1, confusion_matrix(y_, y_pred)))
            print('Recall (test)', recall(1, confusion_matrix(y_, y_pred)))

        if name_model == 'uajfiaoispu':
            print(y_pred)
            print('\nTesting set: \n', confusion_matrix(y_, y_pred))
            print('\nFull set: \n', confusion_matrix(y_full, ypred))
            print('\nPrecision ', precision(1, confusion_matrix(y_, y_pred)))
            print('Recall', recall(1, confusion_matrix(y_, y_pred)))

        # y_pred = np.where(y_pred==-1,0,y_pred)
        # ypred = np.where(ypred==-1,0,ypred)

    pr = precision(1, confusion_matrix(y_, y_pred))
    re = recall(1, confusion_matrix(y_, y_pred))

    prfull = precision(1, confusion_matrix(y_full, ypred))
    refull = recall(1, confusion_matrix(y_full, ypred))

    auc = roc_auc_score(y_, y_pred)
    f1 = f1_score(y_full, ypred)
    f1_ = f1_score(y_, y_pred)

    roc = {'metalearners': name_model, 'precision': pr, 'recall': re, 'auc': auc, 'f1': f1, 'f1_': f1_,
           'prfull': prfull, 'refull': refull}
    warnings.filterwarnings("default")
    return roc, ypred, y_pred, y_prob_full


def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def meta_learner(data1, models, prob):
    """
    input: level 1 data
    outout: roc table
    """
    roc_table = pd.DataFrame(columns=['metalearners', 'precision', 'recall', 'auc', 'f1', 'f1_', 'prfull', 'refull'])

    # split data trainint and testing
    y = data1['y_out']
    X = data1.drop(['y_out'], axis=1)
    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33, random_state=32)

    # starting ensemble
    e_full = np.zeros(len(y))
    e_pred = np.zeros(len(y_test))
    e = 0

    # Some causes are unknown or labeled as 0
    y_train = [i if np.random.binomial(1, prob, 1)[0] == 1 else 0 for i in y_train]
    y_train = pd.Series(y_train)

    predictions = pd.DataFrame(columns=models)

    for m in models:
        roc, yfull, y_pred, y_prob_full = classification_models(y_train, y_test, X_train, X_test, m, y, X)
        # tp_genes.append(flat_index[np.equal(tp_genes01,1)])
        roc_table = roc_table.append(roc, ignore_index=True)
        # ensemble
        if m == 'adapter' or m == 'upu' or m == 'lr' or m == 'rf' or m == 'nn':
            e_full += yfull
            e_pred += y_pred
            e += 1
            predictions[m] = yfull

    # finishing ensemble
    e_full = np.divide(e_full, e)
    e_pred = np.divide(e_pred, e)
    e_full = [1 if i > 0.5 else 0 for i in e_full]
    e_pred = [1 if i > 0.5 else 0 for i in e_pred]

    # fpr, tpr, _ = roc_curve(y_test,e_pred)
    pr = precision(1, confusion_matrix(y_test, e_pred))
    re = recall(1, confusion_matrix(y_test, e_pred))
    prfull = precision(1, confusion_matrix(np.hstack([y_test, y_train]), e_full))
    refull = recall(1, confusion_matrix(np.hstack([y_test, y_train]), e_full))
    print('Emsemble:\n', confusion_matrix(np.hstack([y_test, y_train]), e_full), 'precision and recall:', pr, re)

    auc = roc_auc_score(y_test, e_pred)
    f1 = f1_score(np.hstack([y_test, y_train]), e_full)
    f1_ = f1_score(y_test, e_pred)
    roc = {'metalearners': 'ensemble', 'precision': pr, 'recall': re, 'auc': auc, 'f1': f1, 'f1_': f1_,
           'prfull': prfull, 'refull': refull}
    roc_table = roc_table.append(roc, ignore_index=True)
    return roc_table, predictions, y_prob_full


def nn_classifier(y_train, X_train, X_test, X_full):
    """
    meta-learner
    """

    # https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/october/test-run-neural-binary-classification-using-pytorch
    class Batcher:
        def __init__(self, num_items, batch_size, seed=0):
            self.indices = np.arange(num_items)
            self.num_items = num_items
            self.batch_size = batch_size
            self.rnd = np.random.RandomState(seed)
            self.rnd.shuffle(self.indices)
            self.ptr = 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.ptr + self.batch_size > self.num_items:
                self.rnd.shuffle(self.indices)
                self.ptr = 0
                raise StopIteration  # exit calling for-loop
            else:
                result = self.indices[self.ptr:self.ptr + self.batch_size]
                self.ptr += self.batch_size
                return result

    # ------------------------------------------------------------
    def akkuracy(model, data_x):
        # data_x and data_y are numpy array-of-arrays matrices
        X = T.Tensor(data_x)
        oupt = model(X)  # a Tensor of floats
        oupt = oupt.detach().float()
        oupt = [1 if i > 0.5 else 0 for i in oupt]
        return oupt

    # ------------------------------------------------------------
    class Net(T.nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.hid1 = T.nn.Linear(size, 16)
            self.oupt = T.nn.Linear(16, 1)
            T.nn.init.xavier_uniform_(self.hid1.weight)
            T.nn.init.zeros_(self.hid1.bias)
            T.nn.init.xavier_uniform_(self.oupt.weight)
            T.nn.init.zeros_(self.oupt.bias)

        def forward(self, x):
            z = T.tanh(self.hid1(x))
            z = T.sigmoid(self.oupt(z))  # necessary
            return z

    size = X_train.shape[1]

    net = Net()

    net = net.train()  # set training mode
    lrn_rate = 0.01
    bat_size = 500
    loss_func = T.nn.BCELoss()  # softmax() + binary CE
    optimizer = T.optim.SGD(net.parameters(), lr=lrn_rate)
    max_epochs = 500
    n_items = 8000
    batcher = Batcher(n_items, bat_size)
    X_test = X_test.values
    X_train = X_train.values
    print('Starting training')

    # count_class_0, count_class_1 = y_train.value_counts()

    # Divide by class
    df_class_0 = pd.DataFrame(X_train[y_train == 0])
    df_class_1 = pd.DataFrame(X_train[y_train == 1])

    df_class_0_under = df_class_0.sample(4000)
    df_class_1_over = df_class_1.sample(4000, replace=True)
    X_train2 = pd.concat([df_class_0_under, df_class_1_over], axis=0)
    X_train2['y'] = np.repeat([0, 1], 4000)
    X_train2 = X_train2.sample(frac=1).reset_index(drop=True)
    # print(X_train2.tail())
    y_train2 = X_train2['y']
    X_train2 = X_train2.drop(['y'], axis=1)

    X_train2 = X_train2.values
    y_train2 = y_train2.values

    for epoch in range(0, max_epochs):
        for curr_bat in batcher:
            X = T.Tensor(X_train2[curr_bat])
            Y = T.Tensor(y_train2[curr_bat])
            optimizer.zero_grad()
            oupt = net(X)
            loss_obj = loss_func(oupt, Y)
            loss_obj.backward()
            optimizer.step()
    print('Training complete \n')
    net = net.eval()  # set eval mode

    y_pred = akkuracy(net, X_test)
    yfull = akkuracy(net, X_full)
    return y_pred, yfull



def data_norm(data1):
    """
    normalized data x- mean/sd
    input: dataset to be normalized
    output: normalized dataset

    """
    data1o = np.zeros(data1.shape)
    data1o[:, -1] = data1.iloc[:, -1]

    for i in range(0, data1.shape[1] - 1):
        nonzero = []
        for j in range(data1.shape[0]):
            if data1.iloc[j, i] != 0:
                nonzero.append(data1.iloc[j, i])
        for j in range(data1.shape[0]):
            if data1.iloc[j, i] != 0:
                data1o[j, i] = (data1.iloc[j, i] - np.mean(nonzero)) / np.sqrt(np.var(nonzero))

    data1o = pd.DataFrame(data1o)
    data1o.index = data1.index
    data1o.columns = data1.columns
    return data1o


class learner_BART():
    def __init__(self, X_train, X_test, y_train, y_test):
        super(learner_BART, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        print('Running BART')

    def Find_Optimal_Cutoff(self, target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------
        list type, with optimal cutoff value
        https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
        """
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf - 0).abs().argsort()[:1]]
        return list(roc_t['threshold'])

    def fit(self, n_trees=50, n_burn=100):
        model = SklearnModel(n_trees=n_trees, n_burn=n_burn, n_chains=1, n_jobs=1)
        model.fit(self.X_train, self.y_train)
        y_train_pred = model.predict(self.X_train)  # [:,0:1000] Make predictions on the train set
        y_test_pred = model.predict(self.X_test)  # [:,0:1000] Make predictions on the train set
        thhold = self.Find_Optimal_Cutoff(self.y_train, y_train_pred)
        y_train_pred01 = [0 if item < thhold else 1 for item in y_train_pred]
        y_test_pred01 = [0 if item < thhold else 1 for item in y_test_pred]
        print('Leaner Evaluation:')
        print('...Training set: F1 - ', f1_score(self.y_train, y_train_pred01))
        print('...... confusion matrix: ', confusion_matrix(self.y_train, y_train_pred01).ravel())

        print('...Testing set: F1 - ', f1_score(self.y_test, y_test_pred01))
        print('...... confusion matrix: ', confusion_matrix(self.y_test, y_test_pred01).ravel())

        return model

    def cate(self):
        # TODO: implement cate
        print('implement')

        fpr, tpr, _ = roc_curve(y_test, predp1)
        auc = roc_auc_score(y_test, predp1)
        roc = {'learners': 'DA',
               'fpr': fpr,
               'tpr': tpr,
               'auc': auc}

class learner_deconfounder_algorithm():
    def __init__(self, X_train, X_test, y_train, y_test, k=5):
        super(learner_deconfounder_algorithm, self).__init__()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        X = np.concatenate([self.X_train, self.X_test], axis=0)
        y = np.concatenate([self.y_train, self.y_test], axis=0)
        self.X = X
        self.y = y
        self.n = self.X.shape[0]
        self.ncol = self.X.shape[1]
        self.k = k
        print('Running DA')

    def fit(self, b=100, holdout_prop=0.2, alpha=0.05):
        """
        Implementation of the Deconfounder Algorthm with Prob PCA and Logistic Regression
        Contains: Prob PCA function, Predictive Check and Outcome Model
        input:
        - colnames or possible causes
        - k: dimension of latent space
        - b: number of bootstrap samples
        - alpha: IC test on outcome model
        output:
        - coef: calculated using bootstrap
        Note: Due to time constrains, only one PPCA is fitted
        """
        x, x_val, holdout_mask = self.daHoldout(holdout_prop)
        print('... Done Holdout')
        w, z, x_gen = self.FM_Prob_PCA(x, True)
        print('... Done PPCA')
        pvalue = self.PredictiveCheck(x_val, x_gen, w, z, holdout_mask)
        low = stats.norm(0, 1).ppf(alpha / 2)
        up = stats.norm(0, 1).ppf(1 - alpha / 2)
        del x_gen
        if 0.1 < pvalue < 0.9:
            print('... Pass Predictive Check: ' + str(pvalue))
            print('... Fitting Outcome Model')
            coef = []
            pca = np.transpose(z)

            # Bootstrap to calculate the coefs
            for i in range(b):
                rows = np.random.choice(self.X_train.shape[0], int(self.X_train.shape[0] * 0.85), replace=False)
                coef_, _ = self.OutcomeModel_LR(pca, rows, roc_flag=False)
                coef.append(coef_)
            coef = np.matrix(coef)
            coef = coef[:, 0:self.X_train.shape[1]]  # ?

            # Building IC
            coef_m = np.asarray(np.mean(coef, axis=0)).reshape(-1)
            coef_var = np.asarray(np.var(coef, axis=0)).reshape(-1)
            coef_z = np.divide(coef_m, np.sqrt(coef_var / b))
            coef_z = [1 if low < c < up else 0 for c in coef_z]  # 1 if significative and 0 otherwise

            # if ROC = TRUE, calculate ROC results and score is just for testing set
            del coef_var, coef, coef_
            # w, z, x_gen = FM_Prob_PCA(train, k, False)
            _, roc = self.OutcomeModel_LR(pca, roc_flag=True)
        else:
            print('Failed on Predictive Check. Suggetions: trying a different K')
            coef_m = []
            coef_z = []
            roc = []

        return np.multiply(coef_m, coef_z), coef_m, roc

    def daHoldout(self, holdout_portion):
        """
        Hold out a few values from train set to calculate predictive check
        """
        n_holdout = int(holdout_portion * self.n * self.ncol)
        holdout_row = np.random.randint(self.n, size=n_holdout)
        holdout_col = np.random.randint(self.ncol, size=n_holdout)
        holdout_mask = (sparse.coo_matrix((np.ones(n_holdout), (holdout_row, holdout_col)), shape=self.X.shape)).toarray()
        # holdout_subjects = np.unique(holdout_row)
        holdout_mask = np.minimum(1, holdout_mask)
        x_train = np.multiply(1 - holdout_mask, self.X)
        x_val = np.multiply(holdout_mask, self.X)
        return x_train, x_val, holdout_mask

    def FM_Prob_PCA(self, x, flag_pred=False, stddv_datapoints=1):
        """
        Factor Model: Probabilistic PCA
        input:
            x: data
            k: size of latent variables
            flag_pred: if values to calculate the predictive check should be saved
        output:
            w and z values, generated sample to predictive check
        """
        # Reference: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_PCA.ipynb
        from tensorflow.keras import optimizers
        import tensorflow as tf
        import tensorflow_probability as tfp
        from tensorflow_probability import distributions as tfd
        #tf.enable_eager_execution()

        def PPCA(stddv_datapoints):
            """
            Calculating sub parts of PPCA
            """
            w = yield Root(tfd.Independent(
                tfd.Normal(loc=tf.zeros([self.n, self.k]),
                           scale=2.0 * tf.ones([self.n, self.k]),
                           name="w"), reinterpreted_batch_ndims=2))
            z = yield Root(tfd.Independent(
                tfd.Normal(loc=tf.zeros([self.k, self.ncol]),
                           scale=tf.ones([self.k, self.ncol]),
                           name="z"), reinterpreted_batch_ndims=2))
            x = yield tfd.Independent(tfd.Normal(
                loc=tf.matmul(w, z),
                scale=stddv_datapoints,
                name="x"), reinterpreted_batch_ndims=2)

        def factored_normal_variational_model():
            qw = yield Root(tfd.Independent(tfd.Normal(
                loc=qw_mean, scale=qw_stddv, name="qw"), reinterpreted_batch_ndims=2))
            qz = yield Root(tfd.Independent(tfd.Normal(
                loc=qz_mean, scale=qz_stddv, name="qz"), reinterpreted_batch_ndims=2))



        x_train = tf.convert_to_tensor(x, dtype=tf.float32)
        # x_train = tf.convert_to_tensor(np.transpose(x), dtype=tf.float32)
        Root = tfd.JointDistributionCoroutine.Root
        # num_datapoints, data_dim = x.shape
        # data_dim, num_datapoints = x_train.shape
        concrete_ppca_model = functools.partial(PPCA, stddv_datapoints=stddv_datapoints)
        model = tfd.JointDistributionCoroutine(concrete_ppca_model)
        w = tf.Variable(np.ones([self.n, self.k]), dtype=tf.float32)
        z = tf.Variable(np.ones([self.k, self.ncol]), dtype=tf.float32)

        target_log_prob_fn = lambda w, z: model.log_prob((w, z, x_train))
        losses = tfp.math.minimize(lambda: -target_log_prob_fn(w, z),
                                   optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                                   num_steps=200)
        qw_mean = tf.Variable(np.ones([self.n, self.k]), dtype=tf.float32)
        qz_mean = tf.Variable(np.ones([self.k, self.ncol]), dtype=tf.float32)
        qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.n, self.k]), dtype=tf.float32))
        qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([self.k, self.ncol]), dtype=tf.float32))


        surrogate_posterior = tfd.JointDistributionCoroutine(factored_normal_variational_model)

        losses = tfp.vi.fit_surrogate_posterior(
            target_log_prob_fn,
            surrogate_posterior=surrogate_posterior,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
            num_steps=400)

        x_generated = []
        if flag_pred:
            for i in range(50):
                _, _, x_g = model.sample(value=surrogate_posterior.sample(1))
                x_generated.append(x_g.numpy()[0])
        w, z = surrogate_posterior.variables
        return w.numpy(), z.numpy(), x_generated



    def PredictiveCheck(self, x_val, x_gen, w, z, holdout_mask):
        """
        calculate the predictive check
        input:
            x_val: observed values
            x_gen: generated values
            w, z: from fm_PPCA
            holdout_mask
        output:
            pvalue from the predictive check
        """
        # Data prep, holdout mask operations
        holdout_mask1 = np.asarray(holdout_mask).reshape(-1)
        x_val1 = np.asarray(x_val).reshape(-1)
        x1 = np.asarray(np.multiply(np.dot(w, z), holdout_mask)).reshape(-1)
        del x_val
        x_val1 = x_val1[holdout_mask1 == 1]
        x1 = x1[holdout_mask1 == 1]
        pvals = np.zeros(len(x_gen))
        for i in range(len(x_gen)):
            holdout_sample = np.multiply(x_gen[i], holdout_mask)
            holdout_sample = np.asarray(holdout_sample).reshape(-1)
            holdout_sample = holdout_sample[holdout_mask1 == 1]
            x_val_current = stats.norm(holdout_sample, 1).logpdf(x_val1)
            x_gen_current = stats.norm(holdout_sample, 1).logpdf(x1)
            pvals[i] = np.mean(np.array(x_val_current < x_gen_current))
        return np.mean(pvals)

    def OutcomeModel_LR(self, pca, rows=None, roc_flag=True):
        """
        outcome model from the DA
        input:
        - x: training set
        - x_latent: output from factor model
        - y01: outcome
        """
        import scipy.stats as st
        model = linear_model.SGDClassifier(penalty='l2', alpha=0.1, l1_ratio=0.01, loss='modified_huber',
                                           fit_intercept=True, random_state=0)
        if roc_flag:
            rows_train = range(self.X_train.shape[0])
            rows_test = range(self.X_train.shape[0] + 1, self.X_train.shape[0] + self.X_test.shape[0]+1)
            assert len(rows_train) == len(self.y_train), "Error training set dimensions"
            assert len(rows_test) == len(self.y_test), "Error testing set dimensions"

            X_train = np.concatenate([self.X_train, pca[rows_train, :]], axis=1)
            X_test = np.concatenate([self.X_test, pca[rows_test, :]], axis=1)

            modelcv = calibration.CalibratedClassifierCV(base_estimator=model,
                                                         cv=5, method='isotonic').fit(X_train, self.y_train)
            coef = []
            y_test_pred = modelcv.predict(X_test)
            y_test_predp = modelcv.predict_proba(X_test)
            y_train_pred = modelcv.predict(X_train)

            y_test_predp1 = [i[1] for i in y_test_predp]
            print('Leaner Evaluation:')

            print('... Training set: F1 - ', f1_score(self.y_train, y_train_pred),
                  sum(y_train_pred), sum(self.y_train))
            print('...... confusion matrix: \n', confusion_matrix(self.y_train, y_train_pred))

            print('... Testing set: F1 - ', f1_score(self.y_test, y_test_pred), sum(y_test_pred), sum(self.y_test))
            print('...... confusion matrix: \n', confusion_matrix(self.y_test, y_test_pred))
            fpr, tpr, _ = roc_curve(self.y_test, y_test_predp1)
            auc = roc_auc_score(self.y_test, y_test_predp1)
            roc = {'learners': 'DA',
                   'fpr': fpr,
                   'tpr': tpr,
                   'auc': auc}
        else:
            x_aug = np.concatenate([self.X_train[rows, :], pca[rows, :]], axis=1)
            model.fit(x_aug, self.y_train[rows])
            coef = model.coef_[0]
            roc = {}
        return coef, roc
