import pandas as pd
import numpy as np
from scipy import interp
import glob
import matplotlib.pyplot as plt

from aux_multiseed_training import getTrainingBands

def getAUC(feats, algo, rcut, her2, parent_loo, parent_main):
    # Main AUC
    train_pCR_means, train_pCR_std = getTrainingBands(parent_main, rcut, 'agnost', her2, whichFeats=feats, whichAlgo=algo)
    auc_imag = train_pCR_means[0]
    auc_std = train_pCR_std[0]
    #print(train_pCR_means, train_pCR_std)

    # Other AUCs
    auc_loo_vec = []
    feat_loo_vec = []
    parent_loo_full = parent_loo+'_r{}_her2agnost_rs1/output_*.txt'.format(rcut)
    #print(parent_loo_full)
    for ii,dfname in enumerate(glob.glob(parent_loo_full)):
        #print(ii,dfname)
        loo_feat = dfname.split('_')[-1][:-4]
        if loo_feat=='50':
            loo_feat = 'median_lymph_KDE_knn_50'
        #print(loo_feat)
        auc_loo, _ = getTrainingBands(parent_loo, rcut, 'agnost', her2, whichFeats='LOO_'+loo_feat, whichAlgo=algo)
        #print('   ',auc_loo)
        auc_loo_vec.append(auc_loo[0])
        feat_loo_vec.append(loo_feat)

    # Sort
    auc_loo_vec = np.array(auc_loo_vec)
    auc_diffs = np.abs(auc_loo_vec-auc_imag)
    auc_signed_diffs = auc_loo_vec-auc_imag
    imp_order = np.argsort(auc_diffs)[::-1]
    auc_zscores = auc_diffs/np.std(auc_diffs)
    auc_signed_zscores = auc_signed_diffs/np.std(auc_diffs)

    sorted_zs = np.array(auc_zscores)[imp_order]
    sorted_signed_zs = np.array(auc_signed_zscores)[imp_order]
    sorted_feats =  np.array(feat_loo_vec)[imp_order]

    # Plot
    plt.figure(figsize=(10,5))
    plt.axhspan(auc_imag-auc_std, auc_imag+auc_std, alpha=0.3)
    plt.axhspan(auc_imag-np.std(auc_diffs), auc_imag+np.std(auc_diffs), color='white', alpha=0.3)
    plt.plot(auc_loo_vec, 'o-r')
    plt.plot([0,len(auc_loo_vec)], [auc_imag, auc_imag], ':b')
    plt.xticks(range(len(feat_loo_vec)), feat_loo_vec, rotation=90, size=18)
    #plt.ylim([0.5,1])
    plt.title('HER2 {}, pCR, {}'.format(her2, algo))
    plt.show()

    return sorted_zs, sorted_feats, sorted_signed_zs

def getImpVec(her2, rcut, feats, df_labels, settings):
    lr_z, lr_feat, lr_s = getAUC(feats, 'lr', rcut, her2, settings['train_loo_folder'], settings['train_parent_folder'])
    rf_z, rf_feat, rf_s = getAUC(feats, 'rf', rcut, her2, settings['train_loo_folder'], settings['train_parent_folder'])
    svc_z, svc_feat, svc_s = getAUC(feats, 'svc', rcut, her2, settings['train_loo_folder'], settings['train_parent_folder'])

    return lr_z, lr_feat, lr_s, rf_z, rf_feat, rf_s, svc_z, svc_feat, svc_s


def getLabels():
    df_labels = pd.read_excel('plotting_labels.xlsx')
    # Drop AllRed and treatment because we don't use them
    df_labels = df_labels[df_labels['Classification']!='TreatmentDummy']
    return df_labels

def formatLabels(df):
    df['DataNum'] = df['Data'].copy()
    df['DataNum'].replace('Clinical', 0, inplace=True)
    df['DataNum'].replace('RNA', 1, inplace=True)
    df['DataNum'].replace('DNA', 2, inplace=True)
    df['DataNum'].replace('Digital Pathology', 3, inplace=True)
    df['DataNum'].replace('Treatment', 4, inplace=True)
    df['ClassNum'] = df['Class'].copy()
    df['ClassNum'].replace('Treatment', 0, inplace=True)
    df['ClassNum'].replace('Staging', 1, inplace=True)
    df['ClassNum'].replace('ER HER2 features', 2, inplace=True)
    df['ClassNum'].replace('Proliferation', 3, inplace=True)
    df['ClassNum'].replace('Immune activation', 4, inplace=True)
    df['ClassNum'].replace('Immune evasion', 5, inplace=True)
    df['Label'].replace('PGR expression','$\it{PGR}$ expression',inplace=True)
    df['Label'].replace('ESR1 expression','$\it{ESR1}$ expression',inplace=True)
    df['Label'].replace('ERBB2 expression','$\it{ERBB2}$ expression',inplace=True)
    df['Label'].replace('PIK3CA mutation status','$\it{PIK3CA}$ mutation status',inplace=True)
    df['Label'].replace('TP53 mutation status','$\it{TP53}$ mutation status',inplace=True)
    df['Label'].replace('T-Cell exclusion score','T-cell exclusion score',inplace=True)
    return df
