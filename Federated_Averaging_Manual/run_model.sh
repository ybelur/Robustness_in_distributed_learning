#!/bin/bash

# Define arguments
CSV_FILE="results/test.csv"
NUM_ROUNDS="50"
EPOCHS="1"
NUM_DATA_POISONED_CLIENTS="1"
SCALE_FACTOR="5"
NUM_MODEL_POISONED_CLIENTS="0"

# Run the Python script with the arguments
python fed_avg_data_and_model_poison.py \
    --csv_file $CSV_FILE \
    --num_rounds $NUM_ROUNDS \
    --epochs $EPOCHS \
    --num_data_poisoned_clients $NUM_DATA_POISONED_CLIENTS \
    --scale_factor $SCALE_FACTOR \
    --num_model_poisoned_clients $NUM_MODEL_POISONED_CLIENTS
