def defineFeatures(whichFeat, her2=0):
    import pickle
    if her2==0:
        fam = pickle.load(open('inputs/featnames.p', 'rb'))
    else:
        raise Exception('Only HER2 agnostic allowed')

    from leaveOneOutFeatures import getLOO
    if 'LOO' in whichFeat:
        feats = getLOO(her2, whichFeat)
        print('Running LOO model for feature importance')
        print(feats)
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

def defineSplits(X,ycateg,random_state):
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
    # CV based on RCB categories
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=int(random_state))
    splits = []
    for (tr,ts) in cv.split(X, ycateg):
        splits.append((tr,ts))
    return splits

def defineTrainingSets(df_train, feats, her2=0):
    if her2!=0:
        df_train = df_train[df_train['HER2.status']==her2].copy()
    df_train_reshuffled = df_train.copy().sample(frac=1, random_state=0).reset_index(drop=True)
    X = df_train_reshuffled[feats].copy()
    ycateg = df_train_reshuffled['RCB.category'].copy()
    yscore = df_train_reshuffled['RCB.score'].copy()
    patID = df_train_reshuffled['Trial.ID'].copy()
    return X, ycateg, yscore, patID

def defineTestSet(df_test, feats, her2=0):
    if her2!=0:
        df_test = df_test[df_test['HER2.status']==her2].copy()
    df_test_reshuffled = df_test.copy().sample(frac=1, random_state=0).reset_index(drop=True)
    X = df_test_reshuffled[feats].copy()
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

def main(whichFeats, her2_str, rcut, random_state):
## SETUP
    from classification_models import optimise_logres_featsel, optimise_SVC_featsel, optimise_rf_featsel, run_all_models, refit_all_models, final_test, test_all_models
    import pandas as pd
    import pickle
    plotStyle()

    print('Running {} models'.format(whichFeats))

## OUTPUT
    f = open('/data/'+'output_'+whichFeats+'.txt', 'a')
    f.write('-----'+whichFeats+'------\n')
    f.close()

## INPUT
    her2 = int(her2_str)
    df_train = pd.read_csv('inputs/training_df.csv')

## DATASET
    feats = defineFeatures(whichFeats, her2=her2)
    Xtrain, ytrainCateg, ytrainScore, ytrainID = defineTrainingSets(df_train, feats, her2=her2)
    splits = defineSplits(Xtrain, ytrainCateg, random_state)

## MODELS
    ## pCR
    ytrain_pCR = defineResponse(df_train, 'pCR', her2=her2)
    pcr_models = run_all_models(Xtrain, ytrain_pCR, splits, float(rcut))
    pcr_refits = refit_all_models(Xtrain, ytrain_pCR, pcr_models, splits, whichFeats, 'pCR', ytrainID)
    with open('/data/'+whichFeats+'_pcr_models.pkl', 'wb') as f:
        pickle.dump(pcr_models, f)
    with open('/data/'+whichFeats+'_pcr_refits.pkl', 'wb') as f:
        pickle.dump(pcr_refits, f)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
