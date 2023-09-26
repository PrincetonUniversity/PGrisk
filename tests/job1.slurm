#!/bin/bash
#SBATCH --job-name=submit_rts_daily_full_alloc   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # how many instances of your command are run, total, across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=72:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=xy3134@princeton.edu

module purge
module load anaconda3/2022.10
module load gurobi/10.0.1
conda activate pgrisk

python submit.py 2020-03-31 365 RTS-GMLC \
    /projects/PERFORM/xyang/data/cost_alloc/RTS-GMLC/pgrisk/big-penalty/ \
    /projects/PERFORM/xyang/data/Scenarios/RTS_GMLC/ 0.05 1000 0.05 daily \
    --num-jobs 50 --load-shed-penalty 1e4 --reserve-shortfall-penalty 1e3 --orfeus
