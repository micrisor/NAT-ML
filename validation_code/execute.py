import os
import run_models

her2 = 1

rseeds = [1,2,3,4,5]
featlist = ['clinical', 'dna', 'clin_rna', 'rna', 'imag', 'chemo']
her2_str = {1:'pos', -1:'neg'}

for rs in rseeds:
    output_name = 'results_her2{}_rs{}'.format(her2_str[her2], rs)
    os.makedirs(output_name)
    for feats in featlist:
        run_models.main(feats, her2, rs)
        os.system('mv *csv *png {}'.format(output_name))
    os.system('mv *txt {}'.format(output_name))
