#!/bin/bash

# Define arguments
NUM_CLIENTS="10"
CSV_FILE="results/test.csv"
NUM_ROUNDS="50"
EPOCHS="3"
NUM_DATA_POISONED_CLIENTS="0"
SCALE_FACTOR="5"
NUM_MODEL_POISONED_CLIENTS="0"
AGGREGATION_TYPE="weighted_mean"

# Run the Python script with the arguments
python -u fed_avg_model_poison_prob.py \
    --num_clients $NUM_CLIENTS \
    --csv_file $CSV_FILE \
    --num_rounds $NUM_ROUNDS \
    --epochs $EPOCHS \
    --num_data_poisoned_clients $NUM_DATA_POISONED_CLIENTS \
    --scale_factor $SCALE_FACTOR \
    --num_model_poisoned_clients $NUM_MODEL_POISONED_CLIENTS \
    --aggregation_type $AGGREGATION_TYPE
