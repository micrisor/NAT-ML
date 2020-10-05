from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, mean_squared_error, precision_score, jaccard_score, fowlkes_mallows_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
import sklearn

import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
from scipy import interp
import seaborn as sns
import pandas as pd

def final_test(X, y, model, label='Response', prefix='someresponse'):

    y_pred = model.predict_proba(X)[:,1]
    yreals = y.values

    #Random
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
            label='Chance', alpha=.3)

    # Precision
    roc_auc = roc_auc_score(y, y_pred)

    # AUC
    fpr, tpr, thresholds = roc_curve(y, y_pred)
    plt.plot(fpr, tpr, color='b',
                label=r'Test ROC (AUC = %0.2f)' % (roc_auc),
                lw=2, alpha=.8)

    plt.legend(prop={'size': 15})
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('test_'+prefix+'_'+label+'_roc.png', format='png', bbox_inches='tight')
    plt.close()
    #plt.show()

    df_roc = pd.DataFrame().from_dict({'fpr':fpr, 'tpr':tpr})
    df_roc.to_csv('roc_'+prefix+'_'+label+'_roc.csv')


    #### Precision-recall curve
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    from inspect import signature
    precision, recall, thresholds = precision_recall_curve(yreals, y_pred)
    rand_perf = (yreals==1).sum()/yreals.shape[0]
    average_precision = average_precision_score(yreals, y_pred)

    step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
    plt.step(recall, precision, color='b', alpha=0.2, where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs, label='AP={:.2f}'.format(average_precision))
    plt.plot([0, 1], [rand_perf, rand_perf], ':k')
    plt.xlim([0,1])
    plt.ylim([0,1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig('test_'+prefix+'_'+label+'_precrecall.png', format='png', bbox_inches='tight')

    df_prec = pd.DataFrame().from_dict({'recall':recall, 'precision':precision})
    df_prec.to_csv('prec_'+prefix+'_'+label+'.csv')

    df_data = pd.DataFrame().from_dict({'yreals':yreals, 'ypred':y_pred})
    df_data.to_csv('ydata_'+prefix+'_'+label+'.csv')

    #### Boxplot
    preds_for_0 = np.array([p for (i,p) in enumerate(y_pred) if yreals[i]==0])
    preds_for_1 = np.array([p for (i,p) in enumerate(y_pred) if yreals[i]==1])
    all_stats = [preds_for_0,preds_for_1]

    sns.boxplot(data=all_stats)
    plt.ylabel('Predicted score')
    plt.xticks([0,1], ['non-'+prefix, prefix], rotation=80)
    plt.savefig('test_'+prefix+'_'+label+'_boxplot.png', format='png', bbox_inches='tight')
    plt.close()
    #plt.show()


    #### Bar plot
    ypreds = np.array(y_pred)
    yreals = np.array(yreals)
    order = np.argsort(-ypreds)
    plt.figure(figsize=(7,4))
    sel_pos = yreals[order]==1
    sel_rest = yreals[order]==0
    plt.bar(np.squeeze(np.argwhere(sel_pos==True)), ypreds[order][sel_pos], color='cornflowerblue')
    plt.bar(np.squeeze(np.argwhere(sel_rest==True)), ypreds[order][sel_rest], color='gainsboro')
    plt.ylabel('Predicted score')
    plt.xlabel('Patients')
    #plt.xlim([-1,40])
    plt.savefig('test_'+prefix+'_'+label+'_barplot.png', format='png', bbox_inches='tight')
    plt.close()
    #plt.show()

    f = open('test_output.txt', 'a')
    f.write('test,{},{},{}\n'.format(prefix,label,roc_auc))
    f.close()

    f = open('predictions.txt', 'a')
    f.write('{} {} '.format(prefix, label))
    for eachy in y_pred:
        f.write('{} '.format(eachy))
    f.write('\n')
    f.close()


def test_all_models(X,y,results,criterion):
    test_result = {}
    for model in results.keys():
        test_result[model] = final_test(X,y,results[model][0],label=model,prefix=criterion)
    return test_result


class AveragingModels(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models

    # we define clones of the original models to fit the data in
    def fit(self, X, y):
        self.models_ = [clone(x.best_estimator_) for x in self.models]

        # Train cloned base models
        for model in self.models_:
            model.fit(X, y)

        return self

    #Now we do the predictions for cloned models and average them
    def predict_proba(self, X):
        predictions_0 = np.column_stack([
            model.predict_proba(X)[:,0] for model in self.models_
        ])

        predictions_1 = np.column_stack([
            model.predict_proba(X)[:,1] for model in self.models_
        ])
        means_0 = np.mean(predictions_0, axis=1)
        means_1 = np.mean(predictions_1, axis=1)
        return np.column_stack([means_0, means_1])

### Custom class inspired by:
### https://stackoverflow.com/questions/25250654/how-can-i-use-a-custom-feature-selection-function-in-scikit-learns-pipeline
### https://ramhiser.com/post/2018-04-16-building-scikit-learn-pipeline-with-pandas-dataframe/
class DropCollinear(BaseEstimator, TransformerMixin):
    def __init__(self, thresh):
        self.uncorr_columns = None
        self.thresh = thresh

    def fit(self, X, y):
        cols_to_drop = []

        # Find variables to remove
        X_corr = X.corr()
        large_corrs = X_corr>self.thresh
        indices = np.argwhere(large_corrs.values)
        indices_nodiag = np.array([[m,n] for [m,n] in indices if m!=n])

        if indices_nodiag.size>0:
            indices_nodiag_lowfirst = np.sort(indices_nodiag, axis=1)
            correlated_pairs = np.unique(indices_nodiag_lowfirst, axis=0)
            resp_corrs = np.array([[np.abs(spearmanr(X.iloc[:,m], y).correlation), np.abs(spearmanr(X.iloc[:,n], y).correlation)] for [m,n] in correlated_pairs])
            element_to_drop = np.argmin(resp_corrs, axis=1)
            list_to_drop = np.unique(correlated_pairs[range(element_to_drop.shape[0]),element_to_drop])
            cols_to_drop = X.columns.values[list_to_drop]

        print(cols_to_drop)

        cols_to_keep = [c for c in X.columns.values if c not in cols_to_drop]
        self.uncorr_columns = cols_to_keep

        return self

    def transform(self, X):
        return X[self.uncorr_columns]

    def get_params(self, deep=False):
        return {'thresh': self.thresh}

### Inspired by: https://stackoverflow.com/questions/29412348/selectkbest-based-on-estimated-amount-of-features/29412485
class SelectAtMostKBest(SelectKBest):
    def _check_params(self, X, y):
        if not (self.k == "all" or 0 <= self.k <= X.shape[1]):
            # set k to "all" (skip feature selection), if less than k features are available
            self.k = "all"
