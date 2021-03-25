import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from ast import literal_eval
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, f1_score
from sklearn.linear_model import LinearRegression
import train as models
from sklearn.model_selection import train_test_split


def roc_table_creation(filenames, modelname):
    """
    Read from roc_table.txt
    From predicted values, create roc_table
    "obs";"pred"
    """
    roc_table = pd.DataFrame(columns=['learners', 'fpr', 'tpr', 'auc'])

    for f in filenames:
        table = pd.read_csv(f, sep=';')
        y_pred = 1 - table['pred']
        y_ = table['obs']
        y_pred01 = [1 if i >= 0.5 else 0 for i in y_pred]
        print('F1:', f1_score(y_, y_pred01), sum(y_pred01), sum(y_))
        fpr, tpr, _ = roc_curve(y_, y_pred)
        auc = roc_auc_score(y_, y_pred)
        name = f.replace('.txt', '')
        name = name.replace('results//', '')
        roc = {'learners': name,
               'fpr': fpr,
               'tpr': tpr,
               'auc': auc}
        roc_table = roc_table.append(roc, ignore_index=True)
    roc_table.to_pickle('results//roc_' + modelname + '.txt')


def roc_plot(roc_table1):
    """
    From filenama with roc table with tp,tn, acc and others, fit the plot
    """
    # https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot
    # roc_table1 = pd.read_csv(filename,delimiter=';')
    # coef_table1 = pd.read_csv('results\\coef_15.txt',delimiter=',')
    # roc_table1.columns = ['learners','fpr','tpr','auc']
    # roc_table1 = pd.read_pickle(filename)
    filename = 'test'
    roc_table1.set_index('learners', inplace=True)

    fig = plt.figure(figsize=(8, 6))

    for i in roc_table1.index:
        plt.plot(roc_table1.loc[i]['fpr'],
                 roc_table1.loc[i]['tpr'])

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 12}, loc='lower right', ncol=1)

    fig.savefig('results//plot2_' + filename.split('//')[-1].split('.')[0] + '.png')


def roc_cevae(file1, file2, nsim, modelname):
    """
    This function will join the predictions, make the average predictions across the predicted values of a same dataset
    Then, the ROC values to create a plot will be saved as output

    input:
        file1: predicted y01
        file2: observed y01
        nsim: number of datasets snp_simulated
        modelname: name of the output file

    output:
        file saved at modelname
    """
    roc_table = pd.DataFrame(columns=['learners', 'fpr', 'tpr', 'auc'])
    letter = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

    for i in range(nsim):
        filename = file1 + str(i)
        filenameo = file2 + str(i)
        files_pred = []
        files_obs = []
        for l in letter:
            check = filename + '_' + l + '.txt'
            if os.path.isfile(check):
                files_pred.append(filename + '_' + l + '.txt')
                files_obs.append(filenameo + '_' + l + '.txt')

        pred, obs = pd.DataFrame({}), pd.DataFrame({})

        for fp, fo in zip(files_pred, files_obs):
            if pred.shape[0] == 0:
                pred = pd.read_pickle(fp)
                obs = pd.read_pickle(fo)
            else:
                aux = pd.read_pickle(fo)
                if sum(aux.iloc[:, 1] == obs.iloc[:, 1]) == obs.shape[0]:
                    pred = pd.concat((pred, pd.read_pickle(fp)), axis=1)
                    obs = pd.concat((obs, pd.read_pickle(fo)), axis=1)
                else:
                    print('Error: tests lines are different')

            if pred.shape != obs.shape:
                print('Error! Shapes are different')
                break

        pred_average = np.mean(pred, axis=1)
        # pred_average  = [1 if i>0.5 else 0 for i in pred_average ]

        fpr, tpr, _ = roc_curve(obs.iloc[:, 0], pred_average)
        auc = roc_auc_score(obs.iloc[:, 0], pred_average)
        roc = {'learners': 'cevae' + str(i),
               'fpr': fpr,
               'tpr': tpr,
               'auc': auc}
        roc_table = roc_table.append(roc, ignore_index=True)
        # print(confusion_matrix(obs.iloc[:,0],pred_average))
    roc_table.to_pickle('results//roc_' + modelname + '.txt')


def roc_plot_all(filenames):
    """
    From filenama with roc table with tp,tn, acc and others, fit the plot
    """
    # https://abdalimran.github.io/2019-06-01/Drawing-multiple-ROC-Curves-in-a-single-plot

    for f in filenames:
        roc_table0 = pd.read_pickle(f)
        if f == filenames[0]:
            roc_table1 = roc_table0
        else:
            roc_table1 = pd.concat([roc_table1, roc_table0], axis=0)

    roc_table1.set_index('learners', inplace=True)

    fig = plt.figure(figsize=(8, 6))

    for i in roc_table1.index:
        label = i.replace('dappcalr', 'da')
        plt.plot(roc_table1.loc[i]['fpr'],
                 roc_table1.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(label, roc_table1.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right', ncol=1)

    plt.show()
    fig.savefig('results//plots_realdata//plot_roc_all_realdata.png')


def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return confusion_matrix[label, label] / col.sum()


def recall(label, confusion_matrix):
    row = confusion_matrix[label, :]
    return confusion_matrix[label, label] / row.sum()


def first_level_asmeta(colb, colda, data1, prob=1):
    """
    Evaluating the learners as causal discovery models

    input:
        colb: columns with continuos values (BART, CEVAE)
        colda: columns where non-causal are 0 (DA)
        data1: simulated dataset
        prob: proportion of known causes
    output:
        roc table
    """
    # Learners
    y = data1['y_out']
    X = data1.drop(['y_out'], axis=1)
    from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
    y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33, random_state=33)

    # BART/CEVAE
    # col = ['bart_all',  'bart_FEMALE',  'bart_MALE' ]
    for i in colb:
        X_train[i] = np.abs(X_train[i])
        X_test[i] = np.abs(X_test[i])
        q = X_train[i].quantile(0.9)
        X_train[i] = [1 if j > q else 0 for j in X_train[i]]
        X_test[i] = [1 if j > q else 0 for j in X_test[i]]

    # DA
    # col = ['dappcalr_15_LGG','dappcalr_15_SKCM','dappcalr_15_all','dappcalr_15_FEMALE','dappcalr_15_MALE']
    for i in colda:
        X_train[i] = [1 if j != 0 else 0 for j in X_train[i]]
        X_test[i] = [1 if j != 0 else 0 for j in X_test[i]]

    roc_table = pd.DataFrame(columns=['metalearners', 'precision', 'recall', 'auc', 'f1', 'f1_'])
    for i in X_train.columns:
        pr = precision(1, confusion_matrix(y_test, X_test[i]))
        re = recall(1, confusion_matrix(y_test, X_test[i]))
        auc = roc_auc_score(y_test, X_test[i])
        f1_ = f1_score(y_test, X_test[i])

        y_full = np.hstack([X_test[i], X_train[i]])
        f1 = f1_score(np.hstack([y_test, y_train]), y_full)

        roc = {'metalearners': i, 'precision': pr, 'recall': re, 'auc': auc, 'f1': f1, 'f1_': f1_}
        roc_table = roc_table.append(roc, ignore_index=True)

    return roc_table


def diversity(columns, level1data, target_name='y_out'):
    """
    calculate the Q statistics - diversity score

    input:
        colb: columns with continuos values (BART, CEVAE)
        colda: columns where non-causal are 0 (DA)
        data1: simulated dataset
    output:
        Q average value among the pairs of classifiers

    """

    # Splitting Learners from target
    # y = level1data[target_name]
    X = level1data.drop([target_name], axis=1)
    X = level1data.drop(['causes'], axis=1)

    # BART
    # col = ['bart_all',  'bart_FEMALE',  'bart_MALE' ]
    for i in columns:
        X[i] = np.abs(X[i])
        q = X[i].quantile(0.9)
        X[i] = [1 if j > q else 0 for j in X[i]]

    # DA
    # col = ['dappcalr_15_LGG','dappcalr_15_SKCM','dappcalr_15_all','dappcalr_15_FEMALE','dappcalr_15_MALE']
    # for i in colda:
    #    X[i] = [1 if j != 0 else 0 for j in X[i]]

    q_ = []
    X = X.to_numpy()
    up = len(columns)
    for i in range(up - 1):
        for j in range(i + 1, up):
            tn, fp, fn, tp = confusion_matrix(X[:, i], X[:, j]).ravel()
            q_ij = (tp * tn - fp * fn) / (tp * tn + fp * fn)
            q_.append(q_ij)

    return np.mean(q_), q_


def pehe_calc(true_cause, pred_cause, name, version, prob):
    """
    calculate pehe for the simualted datasets

    input:
        true_cause: array with the true causal effects
        pred_cause: array with the estiamted causal effects of a given method
        name: method's name
        version: dataset version
        prob: proportion of known causes
    return: dataframe with estimated values

    """
    pehe = [0, 0, 0]
    count = [0, 0, 0]
    for j in range(len(true_cause)):
        if true_cause[j] == 0:
            pehe[0] += pow(true_cause[j] - pred_cause[j], 2)
            count[0] += 1
        else:
            pehe[1] += pow(true_cause[j] - pred_cause[j], 2)
            count[1] += 1

        pehe[2] += pow(true_cause[j] - pred_cause[j], 2)
        count[2] += 1
    pehe_ = {'method': name, 'pehe_noncausal': pehe[0] / count[0],
             'pehe_causal': pehe[1] / count[1], 'pehe_overall': pehe[2] / count[2],
             'version': version, 'prob': prob}
    return pehe_


def simulation_eval(nsim):
    """
    evaluation of the simulatiosn
    input: number of simulated datsets
    output: several files with pehe, diversity, accuracy, f1-score of the meta-learners and learners
    """

    # Create output files
    out_meta = pd.DataFrame(
        columns=['metalearners', 'precision', 'recall', 'auc', 'f1', 'f1_', 'prfull', 'refull', 'version', 'prob'])
    out_metac = pd.DataFrame(
        columns=['metalearners', 'precision', 'recall', 'auc', 'f1', 'f1_', 'prfull', 'refull', 'version', 'prob'])
    out_level0 = pd.DataFrame(columns=['metalearners', 'precision', 'recall', 'auc', 'f1', 'f1_', 'version', 'prob'])
    pehe = pd.DataFrame(columns=['method', 'pehe_noncausal', 'pehe_causal', 'pehe_overall', 'version', 'prob'])

    out_diversity = []
    out_diversity_version = []

    # proportion of known causes
    p = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    for prob in p:

        for i in range(nsim):
            # read level 1 data
            data = pd.read_csv('results\\level1data_sim_' + str(i) + '.txt', sep=';')

            # Meta-learners
            exp1 = models.meta_learner(data.iloc[:, [1, 2, 3]], ['rf', 'lr', 'random', 'upu', 'adapter', 'nn'], prob)
            exp1c = models.meta_learner(data.iloc[:, [1, 4, 3]], ['rf', 'lr', 'random', 'upu', 'adapter', 'nn'], prob)
            exp0 = first_level_asmeta(['cevae'], ['coef'], data.iloc[:, [1, 2, 3]])

            exp1['version'] = str(i)
            exp1c['version'] = str(i)
            exp0['version'] = str(i)

            exp1['prob'] = prob
            exp1c['prob'] = prob
            exp0['prob'] = prob

            qav, q_ = diversity(['cevae'], ['coef'], data.iloc[:, [1, 2, 3]])

            # pre-processing to pehe
            X = np.matrix(data.iloc[:, [1, 4]])
            y = np.array(data.iloc[:, 5])
            y_train, y_test, X_train, X_test = train_test_split(y, X, test_size=0.33, random_state=33)

            # model = sm.Logit(y,X).fit_regularized(method='l1')
            y_train = [i if np.random.binomial(1, prob, 1)[0] == 1 else 0 for i in y_train]
            y_train = pd.Series(y_train)

            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_full = model.predict(X)

            pehe_ = pehe_calc(np.array(data.iloc[:, 5]), np.array(data.iloc[:, 1]), 'CEVAE', str(i), prob)
            pehe = pehe.append(pehe_, ignore_index=True)
            pehe_ = pehe_calc(np.array(data.iloc[:, 5]), np.array(data.iloc[:, 2]), 'DA', str(i), prob)
            pehe = pehe.append(pehe_, ignore_index=True)
            pehe_ = pehe_calc(np.array(data.iloc[:, 5]), y_full, 'Meta-learner (Full set)', str(i), prob)
            pehe = pehe.append(pehe_, ignore_index=True)
            pehe_ = pehe_calc(y_test, y_pred, 'Meta-learner (Testing set)', str(i), prob)
            pehe = pehe.append(pehe_, ignore_index=True)

            out_meta = out_meta.append(exp1, ignore_index=True)
            out_metac = out_metac.append(exp1c, ignore_index=True)
            out_level0 = out_level0.append(exp0, ignore_index=True)
            out_diversity.append(qav)
            out_diversity_version.append(i)

    # saving outputs
    diversity_ = pd.DataFrame({'diversity': out_diversity, 'version': out_diversity_version})
    out_meta.to_csv('results\\eval_sim_metalevel1_prob.txt', sep=';')
    out_metac.to_csv('results\\eval_sim_metalevel1c_prob.txt', sep=';')
    out_level0.to_csv('results\\eval_sim_metalevel0_prob.txt', sep=';')
    diversity_.to_csv('results\\eval_sim_diversity_prob.txt', sep=';')
    pehe.to_csv('results\\eval_sim_pehe_prob.txt', sep=';')
