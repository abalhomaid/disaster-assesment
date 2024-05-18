#!/bin/bash -l

#SBATCH -J DOMAIN_BED_METRICS  #name of the job
#SBATCH -o output_metrics.txt   #standard output file
#SBATCH -p gpu-all      #queue used
#SBATCH --gres gpu:v100nv_32GB:1
#SBATCH -c 4            #number of CPUs needed, default is 1
#SBATCH --mem 8000MB    #amount of memory needed, default is 4096 MB per core

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=abalhomaid@hbku.edu.qa


module load slurm cuda11.3/toolkit
conda activate domainbed
srun python model_metrics.py
srun python format_output.py