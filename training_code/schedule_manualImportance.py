def submission(feats, her2word, her2num, rcut, rs, stamp):
    script='''#!/bin/bash
#SBATCH -J results_{stamp}_{feats}_her2{her2word}_r{rcut}_r{rs}
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=mireia.crispinortuzar@cruk.cam.ac.uk
#SBATCH -p general
#SBATCH -o logs/slurm_{stamp}_{feats}_her2{her2word}_r{rcut}_rs{rs}.out
#SBATCH -e logs/slurm_{stamp}_{feats}_her2{her2word}_r{rcut}_rs{rs}.err

singularity exec -H /home/crispi01/projects/breast/response/MolecularData_Classification:/home -B /scratcha/fmlab/crispi01/projects/breast/MolecularData_Classification/results_{stamp}_r{rcut}_her2{her2word}_rs{rs}:/data pythonenv.img python run_models.py {feats} {her2num} {rcut} {rs}
'''.format(feats=feats, her2num=her2num, her2word=her2word, stamp=stamp, rcut=rcut, rs=rs)
    return script

def main():
    import time
    #her2_status = ['neg', 'pos']
    her2_status = ['agnost']
    rcuts = [0.8]
    random_status = [1,2,3,4,5]
    her2_dic = {'neg':-1, 'pos':1, 'agnost':0}
    parameters = [her2_status, rcuts, random_status]

    import os, itertools, time
    parameter_combinations = list(itertools.product(*parameters))

    import datetime
    stamp = 'submission_{date:%Y%m%d_%H%M%S}'.format( date=datetime.datetime.now() )

    explanation = input('What is this submission about? \n')
    flog = open('submissions/log_'+stamp+'.txt', 'w')
    flog.write(explanation)
    flog.close()

    from leaveOneOutFeatures import listLOO
    for i,combi in enumerate(parameter_combinations):
        her2_i, rcut_i, rs_i = combi
        her2_i_num = her2_dic[her2_i]
        print(her2_i_num)
        all_feats = listLOO(her2_i_num)
        for feat_i in all_feats:
            outdir = 'results_{}_r{}_her2{}_rs{}'.format(stamp, rcut_i, her2_i, rs_i)

            if not os.path.exists('/scratcha/fmlab/crispi01/projects/breast/MolecularData_Classification/{}'.format(outdir)):
                os.makedirs('/scratcha/fmlab/crispi01/projects/breast/MolecularData_Classification/{}'.format(outdir))
                print('Making: /scratcha/fmlab/crispi01/projects/breast/MolecularData_Classification/{}'.format(outdir))

            script = submission(feat_i, her2_i, her2_i_num, rcut_i, rs_i, stamp)

            scriptName = 'submissions/submit_{}_{}_{}_her2{}_rs{}'.format(stamp, feat_i, rcut_i, her2_i, rs_i)
            print(' Script name: '+scriptName)

            f = open(scriptName, 'w')
            f.write(script)
            f.close()

            print('  Running: sbatch '+scriptName)
            os.system('sbatch '+scriptName)
            time.sleep(0.2)

if __name__=='__main__':
    main()
