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
from copy import deepcopy


def optimise_logres_featsel(X, y, cut, cv, label='Response', prefix='someresponse', metric='roc_auc'):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    logres = LogisticRegression(random_state=1, penalty='elasticnet', solver='saga', max_iter=10000, n_jobs=-1, class_weight=True)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('logres', logres)])
    # Parameter ranges
    param_grid = { 'kbest__k': np.arange(2,X.shape[1],1),
                    'logres__C': np.logspace(-3,3,30),
                    'logres__l1_ratio': np.arange(0.1,1.1,0.1) }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, iid=False, cv=cv, scoring=metric, return_train_score=True, n_jobs=-1, verbose=0, n_iter=1000, random_state=0)
    search.fit(X,y)

    return search


def optimise_SVC_featsel(X, y, cut, cv=5, label='Response', prefix='someresponse'):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    svc = SVC(random_state=1, max_iter=-1, probability=True)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('svc', svc)])

    param_grid = { 'kbest__k': np.arange(2,X.shape[1],1),
                    'svc__kernel': ['rbf','sigmoid','linear'],
                    'svc__gamma': np.logspace(-9,-2,60),
                    'svc__C': np.logspace(-3,3,60)}

    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, iid=False, cv=cv, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0, n_iter=1000, random_state=0)
    search.fit(X,y)

    return search


def optimise_rf_featsel(X, y, cut, cv=5, label='Response', prefix='someresponse'):
    # Pipeline components
    scaler = StandardScaler()
    kbest = SelectAtMostKBest(score_func=f_classif)
    dropcoll = DropCollinear(cut)
    rf = RandomForestClassifier(random_state=1)
    pipe = Pipeline(steps=[('dropcoll', dropcoll), ('scaler', scaler), ('kbest', kbest), ('rf', rf)])
    # Parameter ranges
    param_grid = { 'kbest__k': range(1,X.shape[1]),
                    "rf__max_depth": [3, None],
                    "rf__n_estimators": [5, 10, 25, 50, 100],
                    "rf__max_features": [0.05, 0.1, 0.2, 0.5, 0.7],
                    "rf__min_samples_split": [2, 3, 6, 10, 12, 15]
                    }
    # Optimisation
    search = RandomizedSearchCV(pipe, param_grid, iid=False, cv=cv, scoring='roc_auc',return_train_score=True, n_jobs=-1, verbose=0,n_iter=1000, random_state=1)
    search.fit(X,y)

    return search


def plot_and_refit(X, y, model, cv, label='Response', prefix='someresponse',feats='features',ids=None):
    aucs = []
    ypreds = []
    yreals = []
    ypreds_cv = []
    yreals_cv = []

    mses = []
    tprs = []
    mean_fpr = np.linspace(0, 1, 10)

    fout = open('/data/'+'patpreds_cv_'+feats+'.txt', 'a')
    cv_models = []
    for i,(tr,ts) in enumerate(cv):
        model.fit(X.iloc[tr,:], y.iloc[tr])
        cv_models.append(deepcopy(model))
        y_pred = model.predict_proba(X.iloc[ts,:])[:,1]
        ytest = y.iloc[ts]
        patID_ytest = ids.iloc[ts]

        # Precision
        ypreds.extend(y_pred)
        yreals.extend(ytest)
        ypreds_cv.append(y_pred)
        yreals_cv.append(ytest)
        roc_auc = roc_auc_score(ytest, y_pred)
        aucs.append(roc_auc)

        # AUC
        fpr, tpr, thresholds = roc_curve(ytest, y_pred)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        ##plt.plot(fpr, tpr, lw=1, alpha=0.3,
                    #label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

        ## Write-up the predictions
        for eachtestpat,eachpred in enumerate(y_pred):
            fout.write('{},{},{},{},{}\n'.format(patID_ytest.values[eachtestpat],feats,prefix,label,eachpred))
    fout.close()

    # Mean curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    median_auc = np.median(aucs)
    std_auc = np.std(aucs)
    #plt.plot(mean_fpr, mean_tpr, color='b',
                #label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                #lw=2, alpha=.8)

    # Error bands
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         #label=r'$\pm$ 1 std. dev.')

    #plt.legend(prop={'size': 15})
    #plt.xlabel('FPR')
    #plt.ylabel('TPR')
    #plt.savefig('/data/cv_'+prefix+'_'+feats+'_'+label+'_roc.png', format='png', bbox_inches='tight')
    #plt.close()
    ##plt.show()

    #f = open('/data/output.txt', 'a')
    f = open('/data/'+'output_'+feats+'.txt', 'a')
    f.write('{},cv,{},{},{},{},{}\n'.format(feats,prefix,label,mean_auc,median_auc,std_auc))
    print('{},cv,{},{},{}\n'.format(feats,prefix,label,mean_auc))
    f.close()


    ### Refit
    model.fit(X,y)
    return [model, cv_models]


def run_all_models(X,y,splits,cut):
    logres_result_auc = optimise_logres_featsel(X,y,cut=cut,cv=splits,metric='roc_auc')
    svc_result = optimise_SVC_featsel(X,y,cut=cut,cv=splits)
    rf_result = optimise_rf_featsel(X,y,cut=cut,cv=splits)
    averaged_models = AveragingModels(models = (logres_result_auc,svc_result,rf_result))
    results = {}
    results['lr'] = logres_result_auc
    results['svc'] = svc_result
    results['rf'] = rf_result
    results['avg'] = averaged_models
    return results

def refit_all_models(X,y,results,splits,whichFeats,criterion,patID=None):
    refit = {}
    for model in results.keys():
        try:
            refit[model] = plot_and_refit(X,y,results[model].best_estimator_,splits,label=model,prefix=criterion,feats=whichFeats,ids=patID)
        except:
            refit[model] = plot_and_refit(X,y,results[model],splits,label=model,prefix=criterion,feats=whichFeats,ids=patID)
    return refit


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
