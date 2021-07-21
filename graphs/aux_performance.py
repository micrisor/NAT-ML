import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample


def getPreds(csv_file, model, feats, algo, rcut):
    df = pd.read_csv(csv_file, delimiter=' ', header=None)
    df = df.iloc[:, :-1]
    truths = df[(df.loc[:,0]==model) & (df.loc[:,1]=='resp')].values[0][2:]
    preds = (df[(df.loc[:,0]=='{}_{}_r{}'.format(model, feats,rcut)) & (df.loc[:,1]==algo)]).values[0][2:]
    preds = np.array([float(x) for x in preds if x is not np.nan])
    truths = np.array([int(x) for x in truths if x is not np.nan])
    return preds, truths

def testBands(testfiles, algo, rcut, her2, feats='rna', model='pCR'):
    if her2=='agnost':
        preds = []
        truths = []
        for her2_i in ['neg', 'pos']:
            outputs_name = testfiles[her2_i]+'/predictions.txt'
            preds_i, truths_i = getPreds(outputs_name, model, feats, algo, rcut)
            preds.extend(preds_i)
            truths.extend(truths_i)
        preds = np.array(preds)
        truths = np.array(truths)
    else:
        outputs_name = testfiles[her2]+'/predictions.txt'
        preds, truths = getPreds(outputs_name, model, feats, algo, rcut)

    nom_auc = roc_auc_score(truths, preds)

    # Bootstrapping
    indices = range(len(truths))
    bs_aucs = []
    for i in range(100):
        bs = resample(indices, replace=True, random_state=i, stratify=truths)
        auc = roc_auc_score(truths[bs], preds[bs])
        bs_aucs.append(auc)
    return np.mean(bs_aucs), np.std(bs_aucs), nom_auc

def getWholeTestBand(testfiles, rcut,her2,model,noBev=False):
    means = []
    sds = []
    noms = []
    for feats in ['clinical','dna','clin_rna','rna','imag','chemo']:
        m,sd,nom = testBands(testfiles, 'avg',rcut,her2,feats=feats,model=model)
        means.append(m)
        sds.append(sd)
        noms.append(nom)
    return np.array(means), np.array(sds),  np.array(noms)

def plotOnlyPCR(train_dfname, trainpath, test_dfname, ext_testfiles, algo, rcut, her2, random_state, prefix=''):
    import os
    from aux_training import getTrainingBands

    testfiles = {}
    testfiles['neg'] = ext_testfiles['neg']+'_rs{}'.format(random_state)
    testfiles['pos'] = ext_testfiles['pos']+'_rs{}'.format(random_state)

    # Train
    try:
        df_train = pd.read_csv(train_dfname, header=None, names=['class','eval','model','algo','mean_auc','med_auc','std_auc'])
    except:
        df_train = train_dfname
    df_train_avg = df_train[df_train['algo']==algo]
    csv_train_pCR_means = df_train_avg.loc[df_train_avg['model']=='pCR', 'mean_auc'].values
    csv_train_pCR_std = df_train_avg.loc[df_train_avg['model']=='pCR', 'std_auc'].values

    train_pCR_means, train_pCR_std = getTrainingBands(trainpath, rcut, 'agnost', her2, random_state)

    print('CSV Train band means:', csv_train_pCR_means)
    print('CSV Train band std:', csv_train_pCR_std)
    print('Train band means:', train_pCR_means)
    print('Train band std:', train_pCR_std)

    # Test
    try:
        # Means
        df_test = pd.read_csv(test_dfname+'test_output.txt', header=None, names=['dataset','props','algo','auc'])
        df_test['resp'] = [x.split('_')[0] for x in df_test['props'].values]
        df_test['feats'] = [x.split('_')[1] for x in df_test['props'].values]
        df_test['rcut'] = [float(x[-3:]) for x in df_test['props'].values]
        df_test_avg = df_test[df_test['algo']=='avg']
        test_pCR_means = df_test_avg.loc[(df_test_avg['resp']=='pCR') & (df_test_avg['rcut']==rcut), 'auc'].values
        print('pCR from output.txt:', test_pCR_means)
    except:
        pass

    test_pCR_bandmeans, test_pCR_bandsds, test_pCR_bandnoms = getWholeTestBand(testfiles, rcut,her2,'pCR')

    print('pCR from band noms:', test_pCR_bandnoms)
    print('pCR from band means:', test_pCR_bandmeans)

    #
    fig0, ax0 = plt.subplots(1,1,figsize=(7,5))
    ax0.plot(range(6), train_pCR_means, ':', color='#20A39E')
    ax0.fill_between(range(6), train_pCR_means+train_pCR_std, train_pCR_means-train_pCR_std, alpha=0.1, color='#20A39E', label='Training')
    ax0.set_ylim([0.45,0.99])
    ax0.set_ylabel('AUC')
    ax0.set_title('HER2{} pCR'.format(her2))
    plt.setp(ax0.get_xticklabels(), visible=True)
    plt.setp(ax0, xticks=range(6), xticklabels=['\nClinical','+\nDNA','+\nRNA','+\nDNA\nRNA','+\nDNA\nRNA\nDigPath','All+\nChemo'])

    ax0.plot(range(6), test_pCR_bandmeans, 'o-', color='#20A39E', label='External validation (BS mean)')
    ax0.plot(range(6), test_pCR_bandnoms, '--', color='#20A39E', label='External validation (Nominal)')
    ax0.plot(range(6), test_pCR_bandmeans+test_pCR_bandsds, '-', color='#20A39E', alpha=0.9)
    ax0.plot(range(6), test_pCR_bandmeans-test_pCR_bandsds, '-', color='#20A39E', alpha=0.9)

    ax0.legend(loc='lower right', fontsize=14)
    #plt.savefig(output_folder+'/performance_feats_her2{}_single_{}_{}A.pdf'.format(her2,algo,prefix), bbox_inches='tight')
    #plt.show()
    #plt.close()

    return train_pCR_means, test_pCR_bandnoms

def join_outputs(folder):
    import glob
    all_files = glob.glob(folder)
    li = []
    for filename in all_files:
        df = pd.read_csv(filename, header=None, names=['class','eval','model','algo','mean_auc','med_auc','std_auc'])
        li.append(df)
    frame = pd.concat(li, axis=0, ignore_index=True)
    frame = frame[frame['class']!='clin_chemo'].copy()

    sorter = ['clinical','dna','clin_rna','rna','imag','chemo']
    # Create the dictionary that defines the order for sorting
    sorterIndex = dict(zip(sorter,range(len(sorter))))
    # Generate a rank column that will be used to sort
    # the dataframe numerically
    frame['class_rank'] = frame['class'].map(sorterIndex)
    frame.sort_values(by='class_rank', inplace = True)
    frame.drop('class_rank', 1, inplace = True)

    return frame
