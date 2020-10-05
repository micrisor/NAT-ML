def defineFeatures(whichFeat, her2=0):
    import pickle
    fam = pickle.load(open('../inputs/transneo_analysis_featnames_extralimited_nochemo1.p', 'rb'))

    if whichFeat == 'clinical':
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

def writeResponse(yreal, yid, prefix):
    f = open('predictions.txt', 'a')

    f.write(prefix+'resp ')
    for eachy in yreal:
        f.write('{} '.format(eachy))
    f.write('\n')

    f.write(prefix+'ID ')
    for eachy in yid:
        f.write('{} '.format(eachy))
    f.write('\n')

    f.close()

def defineTestSet(df_test, feats, her2=0, returnTrialID=False):
    if her2!=0:
        df_test = df_test[df_test['HER2.status']==her2].copy()
        #feats = [x for x in feats if x not in ['HER2.status','ERHER2.status']]
    df_test_reshuffled = df_test.copy().sample(frac=1, random_state=0).reset_index(drop=True)
    X = df_test_reshuffled[feats].copy()
    if returnTrialID==True:
        return X, df_test_reshuffled['Trial.ID'].copy()
    return X

def defineResponse(df,criterion,her2=0):
    if her2!=0:
        df = df[df['HER2.status']==her2].copy()
    df_reshuffled = df.copy().sample(frac=1, random_state=0).reset_index(drop=True)
    y = df_reshuffled['resp.'+criterion].copy()
    return y

def plotStyle():
    from matplotlib import rcParams
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['.Helvetica Neue DeskInterface']
    rcParams['font.size'] = 18
    rcParams['axes.linewidth'] = 2
    rcParams['grid.linewidth'] = 2
    rcParams['grid.color'] = 'gainsboro'
    rcParams['font.weight'] = 'normal'
    rcParams['axes.labelweight'] = 'bold'
    rcParams['axes.labelsize'] = 21
    rcParams['legend.edgecolor'] = 'none'
    rcParams["axes.spines.right"] = False
    rcParams["axes.spines.top"] = False

def removeBev(df):
    new_df = df[df['Chemo.Bev']!=True].reset_index(drop=True).copy()
    return new_df

def artemisOnly(df):
    new_df = df[df['Trial.ID'].str.contains('A')].reset_index(drop=True).copy()
    return new_df

def PBCPOnly(df):
    new_df = df[df['Trial.ID'].str.contains('T')].reset_index(drop=True).copy()
    return new_df


def main(whichFeats, her2_str, random_seed):
## SETUP
    from classification_models import  final_test, test_all_models
    import pandas as pd
    import numpy as np
    import pickle
    plotStyle()

    print('Running {} models, with HER2={} and RS={}'.format(whichFeats, her2_str, random_seed))

## INPUT
    her2 = int(her2_str)
    if her2==-1:
        df_test_pCR = pd.read_csv('../inputs/merged_her2neg_pcr_forvalidation_V3.csv')
    elif her2==1:
        df_test_pCR = pd.read_csv('../inputs/pbcp-her2pos_analysis_dataframe_forvalidation.csv')

    ## Bev cut
    if her2==-1:
        print('HER2={} so removing Bev'.format(her2))
        df_test_pCR = removeBev(df_test_pCR)

    ### Only Artemis / PBCP
    #df_test_pCR = PBCPOnly(df_test_pCR)

## DATASET
    feats = defineFeatures(whichFeats, her2=her2)
    Xtest_pCR, trialID_pCR = defineTestSet(df_test_pCR, feats, returnTrialID=True)

    ## Limited dataset
    parent_folder = 'trained_models/results_submission_20200916_100417'

## MODELS
    ### pCR
    for rcut in [1.0,0.9,0.8,0.7]:
        pcr_refits = pickle.load(open('{}_r{}_her2agnost_rs{}/{}_pcr_refits.pkl'.format(parent_folder, rcut, random_seed, whichFeats), 'rb'))
        ytest_pCR = defineResponse(df_test_pCR, 'pCR')
        writeResponse(ytest_pCR, trialID_pCR, 'pCR ')
        test_all_models(Xtest_pCR, ytest_pCR, pcr_refits, 'pCR_{}_r{}'.format(whichFeats, rcut))

if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])
