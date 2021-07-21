from ipywidgets import interact

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

def fillInFeatures(whichFeat):
    import pandas as pd
    feats = defineFeatures(whichFeat)
    data_dic = {}
    for f in feats:
        a = input(f+':  ')
        data_dic[f] = [a]
    df = pd.DataFrame(data=data_dic)
    return df

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
    X = df_test[feats].copy()
    if returnTrialID==True:
        return X, df_test['Trial.ID'].copy()
    return X

def defineResponse(df,criterion,her2=0):
    y = df['resp.'+criterion].copy()
    return y

def loadPaperValidationData():
    import pandas as pd
    test_file1 = pd.read_csv('../inputs/merged_her2neg_pcr_forvalidation_V3.csv')
    test_file2 = pd.read_csv('../inputs/pbcp-her2pos_analysis_dataframe_forvalidation.csv')
    test_file = pd.concat([test_file1,test_file2], sort=False, ignore_index=True)
    return test_file

def plotStyle():
    from matplotlib import rcParams
    #rcParams['font.family'] = 'sans-serif'
    #rcParams['font.sans-serif'] = ['Arial']
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


def getPreds(whichFeats, df_test_pCR):
## SETUP
    import pandas as pd
    import numpy as np
    import pickle
    import sys

    plotStyle()
    sys.path.append('../validation_code/')

## DATASET
    if isinstance(df_test_pCR,dict):
        for kk in df_test_pCR:
            if not isinstance(df_test_pCR[kk], list):
                df_test_pCR[kk] = [df_test_pCR[kk]]
        df_test_pCR['Trial.ID'] = 'Test000'
        df_test_pCR = pd.DataFrame(df_test_pCR)

    feats = defineFeatures(whichFeats, her2='agnost')
    Xtest_pCR, trialID_pCR = defineTestSet(df_test_pCR, feats, returnTrialID=True)

    try:
        ytest_pCR = defineResponse(df_test_pCR, 'pCR')
    except:
        ytest_pCR = None

## MODEL
    parent_folder = '../trained_models/results_submission_20200916_100417'
    rcut = 0.8

    rseeds = [1,2,3,4,5]
    preds = []
    for random_seed in rseeds:
        modelfile = pickle.load(open('{}_r{}_her2agnost_rs{}/{}_pcr_refits.pkl'.format(parent_folder, rcut, random_seed, whichFeats), 'rb'))
        modelname = 'pCR_{}_r{}'.format(whichFeats, rcut)
        trained_model = modelfile['avg'][0]
        y_pred = trained_model.predict_proba(Xtest_pCR)[:,1]
        preds.append(y_pred)
    preds = np.array(preds)
    rs_avg_preds = np.average(preds, axis=0)

    return rs_avg_preds, ytest_pCR

def plotInteractiveHistogram(**kwargs):
    test_file = kwargs.copy()
    plotHistogram(test_file)

def replaceValues(thedic):
    # Histology
    histo_dic = {'Invasive ductal carcinoma':1, 'Other':0}
    thedic['Histology'] = histo_dic[thedic['Histology']]
    # ER HER2
    posneg_dic = {'Positive':1, 'Negative':-1}
    thedic['ER.status'] = posneg_dic[thedic['ER.status']]
    thedic['HER2.status'] = posneg_dic[thedic['HER2.status']]
    thedic['LN.at.diagnosis'] = posneg_dic[thedic['LN.at.diagnosis']]
    # Others
    yn_dic = {'Yes':1, 'No':0}
    for ff in thedic:
        if isinstance(thedic[ff], str):
            thedic[ff] = yn_dic[thedic[ff]]
    return thedic

def plotHistogram(test_file):
    import matplotlib.pyplot as plt
    import numpy as np

    if isinstance(test_file,dict):
        try:
            test_file = replaceValues(test_file)
        except:
            pass

    # For now this is only available for the fully integrated model
    feats = 'chemo'

    # Get the reference results
    #val_test_file = loadPaperValidationData()
    #val_pred, val_real = getPreds(feats, val_test_file)
    val_pred, val_real = pastePaperValidationData()

    # Get the new results
    new_pred, new_real = getPreds(feats, test_file)

    # Map the colours
    val_col = np.empty_like(val_real, dtype=str)
    val_col[val_real==0] = 'r'
    val_col[val_real==1] = 'g'
    #val_real = val_real.values

    # Sort
    ind = np.argsort(val_pred)[::-1]

    # Plot
    try:
        num_cases = test_file.shape[0]
    except:
        num_cases = 1
    for casenum in range(num_cases):
        # Individual case
        new_x = np.interp(new_pred[casenum], val_pred[ind][::-1], np.arange(len(ind))[::-1])
        new_height = new_pred[casenum]

        # Figure
        plt.figure(figsize=(12,5))
        if num_cases>1:
            plt.title('Case num. {}'.format(casenum+1))
        plt.bar(x=range(len(ind)), height=val_pred[ind],color=val_col[ind])
        plt.bar(x=[0], height=[0], color='g', label='pCR')
        plt.bar(x=[0], height=[0], color='r', label='Residual disease')
        plt.plot(new_x, new_height, '*k', markersize=20, label='Response score={:.2f}'.format(new_height))
        plt.legend()
        plt.xlabel('External validation cases')
        plt.ylabel('Response score')
        plt.show()


def pastePaperValidationData():
    import numpy as np
    preds = np.array([0.22810215,0.40753564,0.25029255,0.24338522,0.58526945,0.47396612,0.21026824,0.22619078,0.16736602,0.48549077,0.51662882,0.40105857,0.4834421,0.14354779,0.37008527,0.30248409,0.2825342,0.40265251,0.16210944,0.10692382,0.05288764,0.56310043,0.0561415,0.13786384,0.23145517,0.08611188,0.07471854,0.2176536,0.22378712,0.09984622,0.07006702,0.09950498,0.14381672,0.13519067,0.07432762,0.42748817,0.12579908,0.06953039,0.11550878,0.41704588,0.06764031,0.08143026,0.3217814,0.06160175,0.07519025,0.11082459,0.0821571,0.08999902,0.22619545,0.2151861,0.13171382,0.1819837,0.33060446,0.09025221,0.06517306,0.11392172,0.21140586,0.53348921,0.48671408,0.51016026,0.44311463,0.56631977,0.27647354,0.25365111,0.17421471,0.44654186,0.33275343,0.09971859,0.3904993,0.24810486,0.22957421,0.55797706,0.69134801,0.21399189,0.19331935])
    reals = np.array([0,1,0,0,0,0,0,0,0,0,1,0,1,0,1,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,1,1,0,1,0,1,0,0,1,1,1,1,1,0,0])
    return preds,reals

