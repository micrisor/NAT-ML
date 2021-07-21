import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import roc_auc_score

modelling_path = '../training_code'
inputs_path = '../inputs'

import sys
sys.path.append(modelling_path)

def getTrainingBands(train_path, rcut, model_her2_str, subset_her2_str, random_state):
    her2dic = {'agnost':0, 'neg':-1, 'pos':1}

    model_her2int = her2dic[model_her2_str]
    subset_her2int = None
    if subset_her2_str is not None:
        subset_her2int = her2dic[subset_her2_str]

    train_dfname = '{}_r{}_her2{}_rs{}'.format(train_path,rcut,model_her2_str,random_state)

    sorter = ['clinical','dna','clin_rna','rna','imag','chemo']
    auc_means = []
    auc_stds = []
    for feats in sorter:
        model_file = pickle.load(open(train_dfname+'/{}_{}_refits.pkl'.format(feats,'pCR'), 'rb'))
        X, y, patID, splits = getData(model_her2int, feats, random_state)
        aucs = []
        for i, (tr,ts) in enumerate(splits):
            y_pred = model_file['avg'][1][i].predict_proba( X.iloc[ts,:] )[:,1]
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


def getData(model_her2_str, whichFeats, random_state):
    from run_models import defineTrainingSets, defineSplits, defineResponse
    model_her2 = int(model_her2_str)
    df_train = pd.read_csv(inputs_path+'/transneo_analysis_dataframe_extralimited_nochemo1.csv')

    ## DATASET
    feats = defineFeatures(whichFeats, her2=model_her2)
    Xtrain, ytrainCateg, ytrainScore, trainID = defineTrainingSets(df_train, feats, her2=model_her2)
    ytrain = defineResponse(df_train, 'pCR', her2=model_her2)
    splits = defineSplits(Xtrain, ytrainCateg, random_state)

    return Xtrain, ytrain, trainID, splits

def defineFeatures(whichFeat, her2=0):
    ### Need to re-define this function to include the longer path to the inputs file
    import pickle
    if her2==0:
        fam = pickle.load(open(inputs_path+'/transneo_analysis_featnames_extralimited_nochemo1.p', 'rb'))
    else:
        raise Exception('You can only run HER2-agnostic models')

    if 'LOO' in whichFeat:
        feats = getLOO(her2, whichFeat)
        #print('Running LOO model for feature importance')
    elif whichFeat == 'clinical':
        feats = fam['clin']
    elif whichFeat == 'dna':
        feats = fam['clin']+fam['dna']
    elif whichFeat == 'rna':
        feats = fam['clin']+fam['dna']+fam['rna']
    elif whichFeat == 'imag':
        feats = fam['clin']+fam['dna']+fam['rna']+fam['digpath']
    elif whichFeat == 'chemo':
        feats = fam['clin']+fam['dna']+fam['rna']+fam['digpath']+fam['chemo']
    elif whichFeat == 'clin_rna':
        feats = fam['clin']+fam['rna']
    elif whichFeat == 'clin_chemo':
        feats = fam['clin']+fam['chemo']
    return feats

def listLOO(her2):
    import pickle
    if her2==0:
        fam = pickle.load(open(inputs_path+'/transneo_analysis_featnames_extralimited_nochemo1.p', 'rb'))
    else:
        raise Exception('You can only run HER2-agnostic models')

    all_feats = fam['clin']+fam['dna']+fam['rna']+fam['digpath']+fam['chemo']
    fam_name = ['LOO_{}'.format(x) for x in all_feats]
    return fam_name

def getLOO(her2, name):
    import pickle
    if her2==0:
        fam = pickle.load(open(inputs_path+'/transneo_analysis_featnames_extralimited_nochemo1.p', 'rb'))
    else:
        raise Exception('You can only run HER2-agnostic models')

    all_feats = fam['clin']+fam['dna']+fam['rna']+fam['digpath']+fam['chemo']

    feat = name.split('_')[1]
    if feat=='median':
        feat='median_lymph_KDE_knn_50'
    loo_feats = [x for x in all_feats if x!=feat]

    return loo_feats
