import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import resample
from inspect import signature
from delong import delong_roc_variance, delong_roc_test
from scipy import stats



def plotMultiPCR(trainpath, ext_testfiles, algo, rcut, her2, prefix='', output_folder=None):
    import os
    from aux_multiseed_training import getTrainingBands

    # Train
    train_pCR_means, train_pCR_std = getTrainingBands(trainpath, rcut, 'agnost', her2)

    print('Train band means:', train_pCR_means)
    print('Train band std:', train_pCR_std)

    ## Test
    test_pCR_bandmeans, test_pCR_bandsds, test_pCR_bandnoms = getWholeTestBand(ext_testfiles, rcut,her2,'pCR')

    print('pCR from band noms:', test_pCR_bandnoms)
    print('pCR from band means:', test_pCR_bandmeans)

    #
    fig0, ax0 = plt.subplots(1,1,figsize=(7,5))
    if her2=='agnost':
        # Draw training bands
        ax0.plot(range(6), train_pCR_means, ':', color='#20A39E')
        ax0.fill_between(range(6), train_pCR_means+train_pCR_std, train_pCR_means-train_pCR_std, alpha=0.1, color='#20A39E', label='Training')
    if her2=='agnost':
        ax0.set_ylim([0.58,0.93])
    ax0.set_ylabel('AUC')
    if her2!='agnost':
        ax0.set_title('HER2{} pCR'.format(her2))
    plt.setp(ax0.get_xticklabels(), visible=True)
    plt.setp(ax0, xticks=range(6), xticklabels=['\nClinical','+\nDNA','+\nRNA','+\nDNA\nRNA','+\nDNA\nRNA\nDigPath','+\nDNA\nRNA\nDigPath\nTreatment'])

    #ax0.plot(range(6), test_pCR_bandmeans, '--', color='#20A39E', label='External validation (BS mean)')
    ax0.plot(range(6), test_pCR_bandnoms, 'o-', color='#20A39E', label='External validation')
    ax0.plot(range(6), test_pCR_bandmeans+test_pCR_bandsds, '-', color='#20A39E', alpha=0.9)
    ax0.plot(range(6), test_pCR_bandmeans-test_pCR_bandsds, '-', color='#20A39E', alpha=0.9)

    ax0.legend(loc='lower right', fontsize=14)

    if output_folder!=None:
        plt.savefig(output_folder+'/aucs_vs_feats_her2{}.pdf'.format(her2), bbox_inches='tight', transparent=True)

    return train_pCR_means, test_pCR_bandnoms


def getPreds(csv_file, model, feats, algo, rcut):
    df = pd.read_csv(csv_file, delimiter=' ', header=None)
    patIds = df.iloc[1, 2:-1].values
    df = df.iloc[:, :-1] # The last column is just nans
    truths = df[(df.loc[:,0]==model) & (df.loc[:,1]=='resp')].values[0][2:]
    preds = (df[(df.loc[:,0]=='{}_{}_r{}'.format(model, feats,rcut)) & (df.loc[:,1]==algo)]).values[0][2:]
    preds = np.array([float(x) for x in preds if x is not np.nan])
    truths = np.array([int(x) for x in truths if x is not np.nan])

    pred_df = pd.DataFrame({'ID':patIds, 'preds':preds, 'truths':truths})
    return pred_df

def testBands(testfiles, algo, rcut, her2, feats='rna', model='pCR'):
    itf = {}
    pred_dfs_forseeds = []
    for random_state in [1,2,3,4,5]:
        itf['neg'] = testfiles['neg']+'_rs{}'.format(random_state)
        itf['pos'] = testfiles['pos']+'_rs{}'.format(random_state)
        if her2=='agnost':
            pred_df_list = []
            for her2_i in ['neg', 'pos']:
                outputs_name = itf[her2_i]+'/predictions.txt'
                pred_df_i = getPreds(outputs_name, model, feats, algo, rcut)
                pred_df_list.append(pred_df_i)
            pred_df = pd.concat(pred_df_list).reset_index()
        else:
            outputs_name = itf[her2]+'/predictions.txt'
            pred_df = getPreds(outputs_name, model, feats, algo, rcut)
        pred_dfs_forseeds.append(pred_df)

    # Average the predictions of the 5 random seeds
    merged_df = pred_dfs_forseeds[0].copy()
    for ii in [1,2,3,4]:
        suffix_df = pred_dfs_forseeds[ii].add_suffix('_rs{}'.format(ii+1))
        merged_df = pd.merge(merged_df, suffix_df, left_on='ID', right_on='ID_rs{}'.format(ii+1))

    series_final_truths = merged_df['truths']
    series_final_preds = (merged_df['preds']+merged_df['preds_rs2']+merged_df['preds_rs3']+merged_df['preds_rs4']+merged_df['preds_rs5'])/5
    merged_df['final_preds'] = series_final_preds

    truths = series_final_truths.values
    preds = series_final_preds.values
    nom_auc = roc_auc_score(truths, preds)

    # Bootstrapping
    indices = range(len(truths))
    bs_aucs = []
    for i in range(100):
        bs = resample(indices, replace=True, random_state=i, stratify=truths)
        auc = roc_auc_score(truths[bs], preds[bs])
        bs_aucs.append(auc)
    return np.mean(bs_aucs), np.std(bs_aucs), nom_auc, [truths,preds]

def getWholeTestBand(testfiles, rcut,her2,model,noBev=False):
    means = []
    sds = []
    noms = []
    for feats in ['clinical','dna','clin_rna','rna','imag','chemo']:
        m,sd,nom,_ = testBands(testfiles, 'avg',rcut,her2,feats=feats,model=model)
        means.append(m)
        sds.append(sd)
        noms.append(nom)
    return np.array(means), np.array(sds),  np.array(noms)

def getROC(testfiles,rcut,her2,model,feats_list,names_list,output_folder=None,suffix=''):
    plt.figure(figsize=(5,5))
    plt.plot([0,1],[0,1],':')
    line_styles = ['--','-','-.',':','-']
    for i,feats in enumerate(feats_list):
        _,_,_,v = testBands(testfiles,'avg',rcut,her2,feats=feats,model=model)
        truths = v[0]
        preds = v[1]
        auc = roc_auc_score(truths, preds)
        label = '{} (Validation AUC={:.2f})'.format(names_list[i], auc)
        fpr, tpr, thresholds = roc_curve(truths, preds)
        plt.plot(fpr, tpr, color='k', lw=2, alpha=.8, linestyle=line_styles[i], label=label)
    plt.legend(fontsize=14, loc='lower right')
    plt.xlabel('FPR')
    plt.ylabel('TPR')

    if output_folder!=None:
        plt.savefig(output_folder+'/roc_her2{}{}.pdf'.format(her2,suffix), bbox_inches='tight', transparent=True)

    plt.show()



########### Precision - recall ##############


from sklearn.metrics import precision_recall_curve, average_precision_score, auc

def makeSinglePrec(theax, truths, preds, label, color):
    precision, recall, thresholds = precision_recall_curve(truths, preds)
    avg_prec = average_precision_score(truths,preds)
    prec_auc = auc(recall, precision)
    print('Avg. prec = {}; AUC = {}'.format(avg_prec, prec_auc))
    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    theax.step(recall, precision, color=color, alpha=0.2, where='post')
    theax.fill_between(recall, precision, alpha=0.2, color=color, **step_kwargs, label=label)

def plotPrecRec(testfiles,rcut,her2,model,feats_list,names_list,output_folder=None):
    fig, ax = plt.subplots(1,1,figsize=(5,5))
    fig.subplots_adjust(hspace=0.4)
    colors = ['r','b']
    for i,feats in enumerate(feats_list):
        _,_,_,v = testBands(testfiles,'avg',rcut,her2,feats=feats,model=model)
        truths = v[0]
        preds = v[1]
        makeSinglePrec(ax, truths, preds, names_list[i], colors[i])
        if i==1:
            rand_perf = (truths==1).sum()/truths.shape[0]
            ax.plot([0, 1], [rand_perf, rand_perf], ':k')
            ax.text(0.01, rand_perf+0.02, 'Random performance', size=12)

    ax.legend(prop={'size': 12})
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    if output_folder!=None:
        plt.savefig(output_folder+'/prec_recall_her2{}.pdf'.format(her2), bbox_inches='tight', transparent=True)

    plt.show()




########### DeLong p-values ############


def getLimits(y_true, y_pred):
    alpha = 0.95
    auc, auc_cov = delong_roc_variance(y_true,y_pred)
    auc_std = np.sqrt(auc_cov)
    lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)
    ci = stats.norm.ppf(lower_upper_q,loc=auc,scale=auc_std)
    ci[ci > 1] = 1
    print('AUC:', auc)
    print('AUC COV:', auc_cov)
    print('95% AUC CI:', ci)
    return [auc,auc_cov,ci]

def plotDeLongCIs(testfiles,rcut,her2,model,feats_list,names_list,output_folder=None,suffix=''):
    plt.figure(figsize=(4,5))
    plt.plot([-0.5,len(names_list)],[0.5,0.5],':k')
    all_truths = []
    all_preds = []
    for i,feats in enumerate(feats_list):
        _,_,_,v = testBands(testfiles,'avg',rcut,her2,feats=feats,model=model)
        truths = v[0]
        preds = v[1]
        auc = getLimits(truths, preds)

        plt.vlines(i, auc[2][0], auc[2][1])
        plt.plot(i, auc[0], 'ok', label=names_list[i], markersize=9)

        all_truths.append(truths)
        all_preds.append(preds)

    # pval
    pval = delong_roc_test(all_truths[0], all_preds[0], all_preds[1])
    thepval = np.power(10,pval[0])[0]
    print(thepval)

    x1, x2 = 0, 1   # columns 'Sat' and 'Sun' (first column: 0, see plt.xticks())
    y, h, col = 1.01, 0.02, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, 'p={:.2f}'.format(thepval), ha='center', va='bottom', color=col)

    plt.ylabel('AUC')
    plt.xlim([-0.5,len(names_list)-0.5])
    plt.text(0.55,0.51,'AUC=0.5',size=13)

    plt.xticks(range(len(names_list)),names_list,rotation=85)

    plt.savefig(output_folder+'/delong_her2{}{}.pdf'.format(her2,suffix), bbox_inches='tight', transparent=True)

    plt.show()
