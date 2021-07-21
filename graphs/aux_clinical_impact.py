from scipy import interp
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score
from inspect import signature
from scipy.interpolate import interp1d
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

from aux_multiseed_performance import testBands

dic_titles = {'pCR':'pCR', 'chemosens':'Chemosens.', 'chemores':'Chemoresistant'}

def getRates(truths,preds,cut):
    num_pos = np.sum(truths)
    num_neg = np.shape(truths)[0]-num_pos
    fpr, tpr, thresholds = roc_curve(truths, preds)
    df = pd.DataFrame({'fpr':fpr,'tpr':tpr,'thresholds':thresholds})

    # Collapse ROC
    avg_df = df.groupby('tpr', as_index=False).min().copy()
    tnr = 1-avg_df['fpr'].values
    fnr = 1-avg_df['tpr'].values
    tn = (tnr)*num_neg/(num_pos+num_neg)*100
    fn = (fnr)*num_pos/(num_pos+num_neg)*100
    thresh = avg_df['thresholds'].values

    # Interpolate
    f = interp1d(fn, tn, 'linear')
    f_rate = interp1d(fn, tnr, 'linear')
    threshold = interp1d(fn, thresh, 'linear')
    tn_optim = f(cut)
    interp_thresh = threshold(cut)

    #plt.plot(fnr,tnr,'k')
    #plt.plot(1-df['tpr'].values, 1-df['fpr'].values, 'x')
    #plt.xlabel('FNR')
    #plt.ylabel('TNR')
    #plt.show()

    return tn_optim

def drawImpactPlots(resp, rcut, testfiles, her2, output_folder='.', ax=None):
    fn_cut = 2

    # Clinical
    _,_,_,v = testBands(testfiles,'avg',rcut,her2,feats='clinical',model='pCR')
    truths = v[0]
    preds = v[1]
    num_pos = np.sum(truths)
    num_neg = np.shape(truths)[0]-num_pos
    num_pos_100 = num_pos/(num_pos+num_neg)*100
    num_neg_100 = num_neg/(num_pos+num_neg)*100
    tn_clin_0 = getRates(truths,preds,0.001)
    tn_clin_x = getRates(truths,preds,fn_cut)

    # Combined
    _,_,_,v2 = testBands(testfiles,'avg',rcut,her2,feats='chemo',model='pCR')
    truths2 = v2[0]
    preds2 = v2[1]
    tn_chemo_0 = getRates(truths2,preds2,0.001)
    tn_chemo_x = getRates(truths2,preds2,fn_cut)

    nr_color = '#ef3b2c'
    nr_color_2 = 'lightcoral'
    ypos = 0
    color = '#20A39E'
    r_color = 'paleturquoise'

    fig, ax = plt.subplots(1, 1, figsize=(6.5,5))
    stack_0 = np.array([num_pos_100, num_pos_100, num_pos_100-fn_cut, num_pos_100, num_pos_100-fn_cut])
    stack_1 = np.array([0, 0, fn_cut, 0, fn_cut])
    stack_2 = np.array([0, tn_clin_0, tn_clin_x, tn_chemo_0, tn_chemo_x])
    stack_3 = 100*np.ones(5)-(stack_0+stack_1+stack_2)
    labels = ['Current \n standard','Clinical \n ML \n FN=0','Clinical \n ML \n FN=2','Integ. \n ML \n FN=0',
                       'Integ. \n ML \n FN=2']

    ax.bar(range(5), stack_0, color=color, label='{}, received NAT'.format(dic_titles[resp]), width=0.7)
    ax.bar(range(5), stack_1, bottom=stack_0, color=r_color, label='{}, did not receive NAT'.format(dic_titles[resp]), width=0.7)
    ax.bar(range(5), stack_2, bottom=stack_0+stack_1, color=nr_color_2, label='RD, spared NAT', width=0.7)
    ax.bar(range(5), stack_3, bottom=stack_0+stack_1+stack_2,color=nr_color, label='RD, received NAT', width=0.7)

    ## Upper annotations
    #ax.text(1.5,110,'0 false neg.',horizontalalignment='center')
    #ax.text(3.5,110,'2 false neg.',horizontalalignment='center')
    #ax.plot([0.7,2.3],[105,105],'-k')
    #ax.plot([2.7,4.3],[105,105],'-k')

    for i in range(1,5):
        improvement = stack_2[i]
        base = stack_0[i]+stack_1[i]
        ax.text(i, base+np.max([2.5, improvement/2-2.5]), '{:.0f}'.format(improvement),
                horizontalalignment='center', color='darkred', weight='bold')

    ax.set_xticks(range(5))
    ax.set_xticklabels(labels)
    ax.set_ylabel('Number of cases')
    plt.legend(bbox_to_anchor=(1,-0.3))

    plt.savefig(output_folder+'/clinical_impact.pdf', bbox_inches='tight', transparent=True)
    plt.show()

