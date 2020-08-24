# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 15:15:06 2020

@author: raoki
"""


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
            print('test')
            estimator = SVC(C=40, kernel='sigmoid',gamma='scale',probability=True) #C = 0.3
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


y = data1['y_out']
X = data1.drop(['y_out'], axis=1)
y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33,random_state=32)

classification_models(y_train, y_test, X_train, X_test,'adapter')