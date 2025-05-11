#!/bin/bash

#PBS -l select=1:ncpus=4:mem=8gb
#PBS -l walltime=72:00:00
#PBS -N dropout_mean_data_poison_batch
#PBS -koed
#PBS -o job_output_files/dropout_mean_data_poison_batch.out.txt
#PBS -e job_output_files/dropout_mean_data_poison_batch.err.txt
cd $PBS_O_WORKDIR

module load tools/prod
module load Python/3.11.3-GCCcore-12.3.0

# Fixed arguments
NUM_CLIENTS="10"
NUM_ROUNDS="50"
EPOCHS="3"
NUM_DATA_POISONED_CLIENTS="0"
SCALE_FACTOR="5"
NUM_MODEL_POISONED_CLIENTS="0"
AGGREGATION_TYPE="dropout_mean"

# Number of times to run the script
NUM_RUNS=5

# Loop to run the script multiple times with different output files
for i in $(seq 1 $NUM_RUNS); do
    CSV_FILE="results/dropout_mean_data_poison_cx3_run${i}.csv"
    echo "Running iteration $i with output: $CSV_FILE"
    
    python -u fed_avg_data_and_model_poison_prob.py \
        --num_clients $NUM_CLIENTS \
        --csv_file $CSV_FILE \
        --num_rounds $NUM_ROUNDS \
        --epochs $EPOCHS \
        --num_data_poisoned_clients $NUM_DATA_POISONED_CLIENTS \
        --scale_factor $SCALE_FACTOR \
        --num_model_poisoned_clients $NUM_MODEL_POISONED_CLIENTS \
        --aggregation_type $AGGREGATION_TYPE
done
