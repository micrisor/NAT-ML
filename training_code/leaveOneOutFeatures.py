def listLOO(her2):
    import pickle
    if her2==0:
        fam = pickle.load(open('inputs/transneo_analysis_featnames_extralimited_nochemo1.p', 'rb'))
    else
        raise Exception('Only HER2 agnostic allowed')

    all_feats = fam['clin']+fam['dna']+fam['rna']+fam['digpath']+fam['chemo']
    fam_name = ['LOO_{}'.format(x) for x in all_feats]
    return fam_name

def getLOO(her2, name):
    import pickle
    if her2==0:
        fam = pickle.load(open('inputs/transneo_analysis_featnames_extralimited_nochemo1.p', 'rb'))
    else
        raise Exception('Only HER2 agnostic allowed')

    all_feats = fam['clin']+fam['dna']+fam['rna']+fam['digpath']+fam['chemo']

    feat = name.split('_')[1]
    if feat=='median':
        feat='median_lymph_KDE_knn_50'
    loo_feats = [x for x in all_feats if x!=feat]

    return loo_feats
