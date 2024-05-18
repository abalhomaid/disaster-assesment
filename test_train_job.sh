#!/bin/bash -l

#SBATCH -J TLLIB  #name of the job
#SBATCH -o output_test_train.txt   #standard output file
#SBATCH -p gpu-all      #queue used
#SBATCH --gres gpu:v100nv_32GB:1
#SBATCH -c 4            #number of CPUs needed, default is 1
#SBATCH --mem 8000MB    #amount of memory needed, default is 4096 MB per core

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=abalhomaid@hbku.edu.qa

module load slurm cuda11.3/toolkit
conda activate tllib
srun export CUDA_VISIBLE_DEVICES=0
# srun python ./examples/domain_adaptation/image_classification/cdan.py data/digits -d Digits -s MNIST -t USPS -a resnet50  --epochs 20
srun python ./examples/domain_adaptation/image_classification/cdan.py data/office31 \
    -d Office31 \
    -s A \
    -t W \
    -a resnet50 \
    --epochs 1 \
    --seed 2 \
    --log logs/cdan/Office31_A2W