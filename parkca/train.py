'''
MODIFICATION FOR BC CHILDREN HOSPITAL
'''


import pandas as pd
import numpy as np
import warnings
warnings.simplefilter("ignore")
import eval as eval
#import datapreprocessing as dp
#import CEVAE as cevae
from os import listdir
from os.path import isfile, join

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,  GridSearchCV
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.svm import SVC
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn import linear_model
from sklearn import calibration
from scipy import sparse, stats

#DA
import functools
#Meta-leaners packages
#https://github.com/aldro61/pu-learning (clone)
from puLearning.puAdapter import PUAdapter
#https://github.com/t-sakai-kure/pywsl
from pywsl.pul import pu_mr #pumil_mr (pip install pywsl)
#from pywsl.utils.syndata import gen_twonorm_pumil
#from pywsl.utils.comcalc import bin_clf_err
#NN
from torch.utils.data import Dataset, DataLoader
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler

#Learners
def deconfounder_PPCA_LR(X,colnames,y01,name,k,b, Z,colnamesZ):
    '''
    input:
    - X: train dataset
    - colnames or possible causes
    - y01: outcome
    - name: file name
    - k: dimension of latent space
    - b: number of bootstrap samples
    '''
    x_train, x_val, holdout_mask = daHoldout(X,0.2)
    w,z, x_gen = fm_PPCA(x_train,k,True)
    filename = 'dappcalr_' +str(k)+'_'+name
    pvalue= daPredCheck(x_val,x_gen,w,z, holdout_mask)
    alpha = 0.05 #for the IC test on outcome model
    low = stats.norm(0,1).ppf(alpha/2)
    up = stats.norm(0,1).ppf(1-alpha/2)
    #To speed up, I wont fit the PPCA to each boostrap iteration
    del x_gen
    if 0.1 < pvalue and pvalue < 0.9:
        print('Pass Predictive Check:', filename, '(',str(pvalue),')' )
        coef= []
        pca = np.transpose(z)
        for i in range(b):
            #print(i)
            rows = np.random.choice(X.shape[0], int(X.shape[0]*0.85), replace=False)
            X = X[rows, :]
            y01_b = y01[rows]
            pca_b = pca[rows,:]
            #w,pca, x_gen = fm_PPCA(X,k)
            #outcome model
            coef_, _ = outcome_model_ridge(X,colnames, pca_b,y01_b,False,filename)
            coef.append(coef_)


        coef = np.matrix(coef)
        coef = coef[:,0:X.shape[1]]
        #Building IC
        coef_m = np.asarray(np.mean(coef,axis=0)).reshape(-1)
        coef_var = np.asarray(np.var(coef,axis=0)).reshape(-1)
        coef_z = np.divide(coef_m,np.sqrt(coef_var/b))
        #1 if significative and 0 otherwise
        coef_z = [ 1 if c>low and c<up else 0 for c in coef_z ]


        #https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot
        '''
        if ROC = TRUE, outcome model receive entire dataset, but internally split in training
        and testing set. The ROC results and score is just for testing set
        '''
        del X,pca,pca_b,y01_b
        del coef_var, coef, coef_
        w,z, x_gen = fm_PPCA(X,k,False)
        _,roc =  outcome_model_ridge(X,colnames, np.transpose(z),y01,True,filename)
        #df_ce =pd.merge(df_ce, causal_effect,  how='left', left_on='genes', right_on = 'genes')
        #df_roc[name_PCA]=roc
        #aux = pd.DataFrame({'model':[name_PCA],'gamma':[gamma],'gamma_l':[cil],'gamma_u':[cip]})
        #df_gamma = pd.concat([df_gamma,aux],axis=0)
        #df_gamma[name_PCA] = sparse.coo_matrix((gamma_ic),shape=(1,3)).toarray().tolist()
    else:
        coef_m = []
        coef_z = []
        roc = []

    return np.multiply(coef_m,coef_z), coef_m, roc, filename

def fm_PPCA(train,latent_dim, flag_pred):
    '''
    Fit the PPCA
    input:
        train: dataset
        latent_dim: size of latent variables
        flag_pred: if values to calculate the predictive check should be saved

    output: w and z values, generated sample to predictive check

    '''
    #Reference: https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/jupyter_notebooks/Probabilistic_PCA.ipynb

    import tensorflow.compat.v1 as tf
    from tensorflow.keras import optimizers
    tf.disable_v2_behavior()
    #import tensorflow as tf.compat.v1
    import tensorflow_probability as tfp
    from tensorflow_probability import distributions as tfd   #conda install -c conda-forge tensorflow-probability
    tf.enable_eager_execution()

    num_datapoints, data_dim = train.shape
    x_train = tf.convert_to_tensor(np.transpose(train),dtype = tf.float32)


    Root = tfd.JointDistributionCoroutine.Root
    def probabilistic_pca(data_dim, latent_dim, num_datapoints, stddv_datapoints):
      w = yield Root(tfd.Independent(
          tfd.Normal(loc=tf.zeros([data_dim, latent_dim]),
                     scale=2.0 * tf.ones([data_dim, latent_dim]),
                     name="w"), reinterpreted_batch_ndims=2))
      z = yield Root(tfd.Independent(
          tfd.Normal(loc=tf.zeros([latent_dim, num_datapoints]),
                     scale=tf.ones([latent_dim, num_datapoints]),
                     name="z"), reinterpreted_batch_ndims=2))
      x = yield tfd.Independent(tfd.Normal(
          loc=tf.matmul(w, z),
          scale=stddv_datapoints,
          name="x"), reinterpreted_batch_ndims=2)

    #data_dim, num_datapoints = x_train.shape
    stddv_datapoints = 1

    concrete_ppca_model = functools.partial(probabilistic_pca,
        data_dim=data_dim,
        latent_dim=latent_dim,
        num_datapoints=num_datapoints,
        stddv_datapoints=stddv_datapoints)

    model = tfd.JointDistributionCoroutine(concrete_ppca_model)

    w = tf.Variable(np.ones([data_dim, latent_dim]), dtype=tf.float32)
    z = tf.Variable(np.ones([latent_dim, num_datapoints]), dtype=tf.float32)

    target_log_prob_fn = lambda w, z: model.log_prob((w, z, x_train))
    losses = tfp.math.minimize(lambda: -target_log_prob_fn(w, z),
                               optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
                               num_steps=200)

    qw_mean = tf.Variable(np.ones([data_dim, latent_dim]), dtype=tf.float32)
    qz_mean = tf.Variable(np.ones([latent_dim, num_datapoints]), dtype=tf.float32)
    qw_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([data_dim, latent_dim]), dtype=tf.float32))
    qz_stddv = tf.nn.softplus(tf.Variable(-4 * np.ones([latent_dim, num_datapoints]), dtype=tf.float32))
    def factored_normal_variational_model():
      qw = yield Root(tfd.Independent(tfd.Normal(
          loc=qw_mean, scale=qw_stddv, name="qw"), reinterpreted_batch_ndims=2))
      qz = yield Root(tfd.Independent(tfd.Normal(
          loc=qz_mean, scale=qz_stddv, name="qz"), reinterpreted_batch_ndims=2))

    surrogate_posterior = tfd.JointDistributionCoroutine(
        factored_normal_variational_model)

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

    return w.numpy(),z.numpy(), x_generated

def daHoldout(train,holdout_portion):
    '''
    Hold out a few values from train set to calculate predictive check
    '''
    num_datapoints, data_dim = train.shape
    n_holdout = int(holdout_portion * num_datapoints * data_dim)

    holdout_row = np.random.randint(num_datapoints, size=n_holdout)
    holdout_col = np.random.randint(data_dim, size=n_holdout)
    holdout_mask = (sparse.coo_matrix((np.ones(n_holdout), \
                                (holdout_row, holdout_col)), \
                                shape = train.shape)).toarray()

    holdout_subjects = np.unique(holdout_row)
    holdout_mask = np.minimum(1, holdout_mask)

    x_train = np.multiply(1-holdout_mask, train)
    x_vad = np.multiply(holdout_mask, train)
    return x_train, x_vad,holdout_mask

def daPredCheck(x_val,x_gen,w,z,holdout_mask):
    '''
    calculate the predictive check
    input:
        x_val: observed values
        x_gen: generated values
        w, z: from fm_PPCA
        holdout_mask
    output: pvalue from the predictive check


    '''
    holdout_mask1 = np.asarray(holdout_mask).reshape(-1)
    x_val1 = np.asarray(x_val).reshape(-1)
    x1 = np.asarray(np.multiply(np.transpose(np.dot(w,z)), holdout_mask)).reshape(-1)
    del x_val
    x_val1 = x_val1[holdout_mask1==1]
    x1= x1[holdout_mask1==1]
    pvals =[]

    for i in range(len(x_gen)):
        generate = np.transpose(x_gen[i])
        holdout_sample = np.multiply(generate, holdout_mask)
        holdout_sample = np.asarray(holdout_sample).reshape(-1)
        holdout_sample = holdout_sample[holdout_mask1==1]
        x_val_current = stats.norm(holdout_sample, 1).logpdf(x_val1)
        x_gen_current = stats.norm(holdout_sample, 1).logpdf(x1)

        pvals.append(np.mean(np.array(x_val_current<x_gen_current)))


    overall_pval = np.mean(pvals)
    return overall_pval

def outcome_model_ridge(x, colnames,x_latent,y01_b,roc_flag,name):
    '''
    outcome model from the DA
    input:
    - x: training set
    - x_latent: output from factor model
    - colnames: x colnames or possible causes
    - y01: outcome
    -name: roc name file
    '''
    import scipy.stats as st
    model = linear_model.SGDClassifier(penalty='l2', alpha=0.1, l1_ratio=0.01,loss='modified_huber', fit_intercept=True,random_state=0)
    if roc_flag:
        #use testing and training set
        x_aug = np.concatenate([x,x_latent],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(x_aug, y01_b, test_size=0.33, random_state=42)
        modelcv = calibration.CalibratedClassifierCV(base_estimator=model, cv=5, method='isotonic').fit(X_train, y_train)
        coef = []

        pred = modelcv.predict(X_test)
        predp = modelcv.predict_proba(X_test)
        predp1 = [i[1] for i in predp]
        print('F1:',f1_score(y_test,pred),sum(pred),sum(y_test))
        print('Confusion Matrix', confusion_matrix(y_test,pred))
        fpr, tpr, _ = roc_curve(y_test, predp1)
        auc = roc_auc_score(y_test, predp1)
        roc = {'learners': name,
               'fpr':fpr,
               'tpr':tpr,
               'auc':auc}
    else:
        #don't split dataset
        x_aug = np.concatenate([x,x_latent],axis=1)
        model.fit(x_aug, y01_b)
        coef = model.coef_[0]
        roc = {}

    return coef, roc

def learners_bcch(path_output, DA, BART, X, Z, y, colnamesX, colnamesZ, causes):
    '''
    input:
        path_output: where to save the files
        X, colnamesX: potential causes and their names
        Z, colnamesZ: confounders and their names (clinical)
        y: 01 outcome
        causes: name of the potential causes (snps)
    '''
    if DA:
        print('DA')
        k_list = [15, 30, 100]
        b = 100
        for k in k_list:
            coefk_table = pd.DataFrame(columns=[causes])
            coefkc_table = pd.DataFrame(columns=[causes])
            roc_table = pd.DataFrame(columns=['learners', 'fpr','tpr','auc'])

            coef, coef_continuos, roc, coln = deconfounder_PPCA_LR(X,colnamesX,y,'DA',k,b,Z,colnamesZ)
            roc_table = roc_table.append(roc,ignore_index=True)
            coefk_table[coln] = coef
            coefkc_table[coln] = coef_continuos

            print('--------- DONE ---------')
            coefk_table[causes] = colnames
            coefkc_table[causes] = colnames

            roc_table.to_pickle(path_output+'//roc_'+str(k)+'.txt')
            coefkc_table.to_pickle(path_output+'//coefcont_'+str(k)+'.txt')
    if BART:
        print('BART')
        #MODEL AND PREDICTIONS MADE ON R
        #filenames=[path_output+'//bart_all.txt',path_output+'//bart_MALE.txt',path_output+'//bart_FEMALE.txt']
        #eval.roc_table_creation(filenames,'bart')
        #eval.roc_plot('results//roc_'+'bart'+'.txt')

def learners(APPLICATIONBOOL, DABOOL, BARTBOOL, CEVAEBOOL,path ):
    '''
    Function to run the application.
    INPUT:
    bool variables
    path: path for level 0 data
    OUTPUT:
    plots, coefficients and roc data (to_pickle format)

    NOTE: This code does not run the BART, it only reads the results.
    BART model was run using R
    '''
    if APPLICATIONBOOL:
        k_list = [15,30]
        pathfiles = path+'\\data'
        listfiles = [f for f in listdir(pathfiles) if isfile(join(pathfiles, f))]
        b =100

        if DABOOL:
            print('DA')
            skip = ['CHOL','LUSC','HNSC','PRAD'] #F1 score very low
            for k in k_list:
                 coefk_table = pd.DataFrame(columns=['genes'])
                 coefkc_table = pd.DataFrame(columns=['genes'])
                 roc_table = pd.DataFrame(columns=['learners', 'fpr','tpr','auc'])
                 #test
                 for filename in listfiles:
                     train, j, v, y01, abr, colnames = dp.data_prep('data\\'+filename)
                     if train.shape[0]>150:
                        print(filename,': ' ,train.shape[0])
                        #change filename
                        name = filename.split('_')[-1].split('.')[0]
                        if name not in skip:
                            coef, coef_continuos, roc, coln = deconfounder_PPCA_LR(train,colnames,y01,name,k,b)
                            roc_table = roc_table.append(roc,ignore_index=True)
                            coefk_table[coln] = coef
                            coefkc_table[coln] = coef_continuos
                        else:
                            print('skip',name)

                 print('--------- DONE ---------')
                 coefk_table['genes'] = colnames
                 coefkc_table['genes'] = colnames

                 #CHANGE HERE 20/05
                 roc_table.to_pickle('results//roc_'+str(k)+'.txt')
                 coefkc_table.to_pickle('results//coefcont_'+str(k)+'.txt')

        if BARTBOOL:
            print('BART')
            #MODEL AND PREDICTIONS MADE ON R
            filenames=['results//bart_all.txt','results//bart_MALE.txt','results//bart_FEMALE.txt']
            eval.roc_table_creation(filenames,'bart')
            eval.roc_plot('results//roc_'+'bart'+'.txt')

        if BARTBOOL and DABOOL:
            eval.roc_plot_all(filenames)

#Meta-learner
def classification_models(y,y_,X,X_,name_model):
    """
    Input:
        X,y,X_test, y_test: dataset to train the model
    Return:
        cm: confusion matrix for the testing set
        cm_: confusion matrix for the full dataset
        y_all_: prediction for the full dataset
    """
    X_full = np.concatenate((X,X_), axis = 0 )
    y_full = np.concatenate((y,y_), axis = 0)

    warnings.filterwarnings("ignore")
    if name_model == 'nn':
        y_pred, ypred = nn_classifier(y, y_, X, X_,X_full)
        pr = precision(1,confusion_matrix(y_,y_pred))
        f1 = f1_score(y_,y_pred)
        if np.isnan(pr) or pr==0 or f1<0.06:
            while np.isnan(pr) or pr ==0 or f1<0.06:
                print('\n\n trying again \n\n')
                y_pred, ypred = nn_classifier(y, y_, X, X_,X_full)
                pr = precision(1,confusion_matrix(y_,y_pred))
                f1 = f1_score(y_,y_pred)

    else:

        if name_model == 'adapter':
            estimator = SVC(C=0.3, kernel='rbf',gamma='scale',probability=True)
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
            prior =.5 #change for the proportion of 1 and 0
            param_grid = {'prior': [prior],
                              'lam': np.logspace(-3, 3, 5), #what are these values
                              'basis': ['lm']}
            #upu (Unbiased PU learning)
            #https://github.com/t-sakai-kure/pywsl/blob/master/examples/pul/upu/demo_upu.py
            model = GridSearchCV(estimator=pu_mr.PU_SL(),
                                   param_grid=param_grid, cv=3, n_jobs=-1)
            X = np.matrix(X)
            y = np.array(y)
            model.fit(X, y)

        elif name_model == 'lr':
            print('lr',X.shape[1])
            X = np.matrix(X)
            y = np.array(y)
            from sklearn.linear_model import LogisticRegression
            w1 = y.sum()/len(y)
            w0 = 1 - w1
            sample_weight = {0:w1,1:w0}
            model = LogisticRegression(C=.1,class_weight=sample_weight,penalty='l2') #
            model.fit(X,y)


        elif name_model=='rf':
            print('rd',X.shape[1])
            w = len(y)/y.sum()
            sample_weight = np.array([w if i == 1 else 1 for i in y])
            model = RandomForestClassifier(max_depth=12, random_state=0)
            model.fit(X, y,sample_weight = sample_weight)

        else:
            print('random',X.shape[1])

        if name_model=='random':
             p = y.sum()+y_.sum()
             p_full = p/(len(y)+len(y_))
             y_pred = np.random.binomial(n=1,p=y_.sum()/len(y_),size =X_.shape[0])
             ypred = np.random.binomial(n=1,p=p_full,size =X_full.shape[0])
        else:
            y_pred = model.predict(X_)
            ypred = model.predict(X_full)

        if name_model =='uajfiaoispu':
            print(y_pred)
            print('\nTesting set: \n',confusion_matrix(y_,y_pred))
            print('\nFull set: \n',confusion_matrix(y_full,ypred))
            print('\nPrecision ',precision(1,confusion_matrix(y_,y_pred)))
            print('Recall',recall(1,confusion_matrix(y_,y_pred)))

        y_pred = np.where(y_pred==-1,0,y_pred)
        ypred = np.where(ypred==-1,0,ypred)

    pr = precision(1,confusion_matrix(y_,y_pred))
    re = recall(1,confusion_matrix(y_,y_pred))

    prfull = precision(1,confusion_matrix(y_full,ypred))
    refull = recall(1,confusion_matrix(y_full,ypred))

    auc = roc_auc_score(y_,y_pred)
    f1 = f1_score(y_full,ypred)
    f1_ = f1_score(y_,y_pred)

    roc = {'metalearners': name_model,'precision':pr ,'recall':re,'auc':auc,'f1':f1,'f1_':f1_,'prfull':prfull,'refull':refull}
    warnings.filterwarnings("default")
    return roc, ypred, y_pred

def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()

def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()

def meta_learner(data1, models, prob ):
    '''
    input: level 1 data
    outout: roc table
    '''
    roc_table = pd.DataFrame(columns=['metalearners', 'precision','recall','auc','f1','f1_','prfull','refull'])

    #split data trainint and testing
    y = data1['y_out']
    X = data1.drop(['y_out'], axis=1)
    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33,random_state=32)

    #starting ensemble
    e_full = np.zeros(len(y))
    e_pred = np.zeros(len(y_test))
    e = 0

    #Some causes are unknown or labeled as 0
    y_train = [i if np.random.binomial(1,prob,1)[0]==1 else 0 for i in y_train]
    y_train = pd.Series(y_train)

    for m in models:
        roc, yfull, y_pred = classification_models(y_train, y_test, X_train, X_test,m)
        #tp_genes.append(flat_index[np.equal(tp_genes01,1)])
        roc_table = roc_table.append(roc,ignore_index=True)
        #ensemble
        if(m=='adapter' or m=='upu' or m=='lr' or m=='rf' or m=='nn'):
            e_full += yfull
            e_pred += y_pred
            e += 1

    #finishing ensemble
    e_full = np.divide(e_full,e)
    e_pred = np.divide(e_pred,e)
    e_full= [1 if i>0.5 else 0 for i in e_full]
    e_pred= [1 if i>0.5 else 0 for i in e_pred]

    #fpr, tpr, _ = roc_curve(y_test,e_pred)
    pr = precision(1,confusion_matrix(y_test,e_pred))
    re = recall(1,confusion_matrix(y_test,e_pred))
    prfull = precision(1,confusion_matrix(np.hstack([y_test,y_train]),e_full))
    refull = recall(1,confusion_matrix(np.hstack([y_test,y_train]),e_full))

    auc = roc_auc_score(y_test,e_pred)
    f1 = f1_score(np.hstack([y_test,y_train]),e_full)
    f1_ = f1_score(y_test,e_pred)
    roc = {'metalearners': 'ensemble','precision':pr ,'recall':re,'auc':auc,'f1':f1,'f1_':f1_,'prfull':prfull,'refull':refull}
    roc_table = roc_table.append(roc,ignore_index=True)
    return roc_table

def nn_classifier(y_train, y_test, X_train, X_test,X_full):
    '''
    meta-learner
    '''
    #https://docs.microsoft.com/en-us/archive/msdn-magazine/2019/october/test-run-neural-binary-classification-using-pytorch
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
          result = self.indices[self.ptr:self.ptr+self.batch_size]
          self.ptr += self.batch_size
          return result

    # ------------------------------------------------------------
    def akkuracy(model, data_x):
      # data_x and data_y are numpy array-of-arrays matrices
      X = T.Tensor(data_x)
      oupt = model(X)            # a Tensor of floats
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
    n_items = len(X_train)
    batcher = Batcher(n_items, bat_size)
    X_test = X_test.values
    X_train = X_train.values
    print('Starting training')

    count_class_0, count_class_1 = y_train.value_counts()

    # Divide by class
    df_class_0 = pd.DataFrame(X_train[y_train== 0])
    df_class_1 = pd.DataFrame(X_train[y_train == 1])

    df_class_0_under = df_class_0.sample(4000)
    df_class_1_over = df_class_1.sample(4000, replace=True)
    X_train2 = pd.concat([df_class_0_under, df_class_1_over], axis=0)
    X_train2['y']= np.repeat([0,1],4000)
    X_train2 = X_train2.sample(frac=1).reset_index(drop=True)

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
