#!/bin/bash -l

#SBATCH -J TLLIB  #name of the job
#SBATCH -o output_train_cdan_n_e.txt   #standard output file
#SBATCH -p gpu-all      #queue used
#SBATCH --gres gpu:1
#SBATCH -c 2            #number of CPUs needed, default is 1
#SBATCH --mem 4096MB    #amount of memory needed, default is 4096 MB per core

#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=abalhomaid@hbku.edu.qa

module load slurm cuda11.3/toolkit
conda activate tllib_metric
srun export CUDA_VISIBLE_DEVICES=0

root="/home/local/QCRI/abalhomaid/projects/disaster/Transfer-Learning-Library"
epochs=50
source="N"
target="E"
phase="analysis"

# model="$root/examples/domain_adaptation/image_classification/cdan.py"
# model="$root/examples/domain_generalization/image_classification/coral.py"
# model="$root/examples/domain_generalization/image_classification/mmd.py"
# model="$root/examples/domain_adaptation/image_classification/dann.py"

models=(
    "$root/examples/domain_adaptation/image_classification/cdan.py" \
    "$root/examples/domain_generalization/image_classification/coral.py" \
    "$root/examples/domain_generalization/image_classification/mmd.py" \
    "$root/examples/domain_adaptation/image_classification/dann.py"
)

for ((m = 0; m < ${#models[@]}; m++)); do
    model="${models[m]}"
    model_name=$(basename "$model")
    model_name="${model_name%.py}"

    if [ "$model_name" == "dann" ]; then
        # DANN, AWDANN
        SWD=(0 0.01)
        trade_offs=(1 1)
        seeds=(32)
    else
        # CDAN, CORAL, MMD
        SWD=(0)
        trade_offs=(1)
        seeds=(32)
    fi

    for ((i = 0; i < ${#SWD[@]}; i++)); do
        current_swd="${SWD[i]}"
        current_trade_off="${trade_offs[i]}"
        for ((j = 0; j < ${#seeds[@]}; j++)); do
            seed="${seeds[j]}"
            echo "Using model ${model} with seed ${seed}"
            echo "Using Wasserstein Distance with SWD = ${current_swd} on ${source} -> ${target} and trade_off = ${current_trade_off}"

        if [ "$model_name" == "mmd" ]; then
            srun python $model \
                "$root/data/damage" \
                -d Damage \
                -s "$source" \
                -t "$target" \
                -a resnet50 \
                --epochs "$epochs" \
                --seed $seed \
                --log "${root}/train_jobs/cdan/logs/${model_name}/seed_${seed}/Damage_${source}2${target}_SWD_${current_swd}_trade_offs_${current_trade_off}" \
                --lambda_s $current_swd \
                --trade-off $current_trade_off \
                --phase $phase \
                --guassian
        else
            srun python $model \
                "$root/data/damage" \
                -d Damage \
                -s "$source" \
                -t "$target" \
                -a resnet50 \
                --epochs "$epochs" \
                --seed $seed \
                --log "${root}/train_jobs/cdan/logs/${model_name}/seed_${seed}/Damage_${source}2${target}_SWD_${current_swd}_trade_offs_${current_trade_off}" \
                --lambda_s $current_swd \
                --trade-off $current_trade_off \
                --phase $phase
        fi

        done
    done
done