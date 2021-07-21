import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score
from aux_training import getData

modelling_path = '../training_code'
inputs_path = '../inputs'

import sys
sys.path.append(modelling_path)

def getTrainingBands(train_path, rcut, model_her2_str, subset_her2_str, whichFeats='iterate', whichAlgo='avg'):
    her2dic = {'agnost':0, 'neg':-1, 'pos':1}

    model_her2int = her2dic[model_her2_str]
    subset_her2int = None
    if subset_her2_str is not None:
        subset_her2int = her2dic[subset_her2_str]


    if whichFeats=='iterate':
        sorter = ['clinical','dna','clin_rna','rna','imag','chemo']
    else:
        sorter = [whichFeats]

    auc_means = []
    auc_stds = []
    for feats in sorter:
        aucs = []
        for random_state in [1,2,3,4,5]:
            train_dfname = '{}_r{}_her2{}_rs{}'.format(train_path,rcut,model_her2_str,random_state)
            model_file = pickle.load(open(train_dfname+'/{}_{}_refits.pkl'.format(feats,'pCR'), 'rb'))
            X, y, patID, splits = getData(model_her2int, feats, random_state)
            for i, (tr,ts) in enumerate(splits):
                y_pred = model_file[whichAlgo][1][i].predict_proba( X.iloc[ts,:] )[:,1]
                y_test = y.iloc[ts]
                if subset_her2int==1 or subset_her2int==-1:
                    her2_statuses = X.iloc[ts,:]['HER2.status'].values
                    y_pred = y_pred[her2_statuses==subset_her2int].copy()
                    y_test = y_test[her2_statuses==subset_her2int].copy()
                nom_auc = roc_auc_score(y_test, y_pred)
                aucs.append(nom_auc)
        auc_means.append( np.mean(aucs) )
        auc_stds.append( np.std(aucs) )
    return np.array(auc_means), np.array(auc_stds)
