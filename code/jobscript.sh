#!/bin/bash
#SBATCH --cpus-per-task=48
#SBATCH --mem=96g
#SBATCH --time=4-00:00 # Runtime in D-HH:MM
########### sbatch --job-name=name --export=var_name=value[,var_name=value...]

module load python/3.8

if [[ ${TASK} == "PS" ]]; then
    python -W ignore 05_1.PanCancer_20Models_HyperParams_Search.py ${MLM} ${RAND}
elif [[ ${TASK} == "PE" ]]; then
    python -W ignore 05_2.PanCancer_20Models_evaluation.py ${MLM} ${RAND}
elif [[ ${TASK} == "NS" ]]; then
    python -W ignore 08_1.NSCLC_20Models_HyperParams_Search.py ${MLM} ${DATA} ${RAND}
elif [[ ${TASK} == "NE" ]]; then
    python -W ignore 08_2.NSCLC_20Models_evaluation.py ${MLM} ${DATA} ${RAND}
else
    echo "TASK code not recognized!"
fi
