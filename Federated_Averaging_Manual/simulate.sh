#!/bin/bash

# Define arrays for rounds and epochs per round
# rounds=(70)
# epochs_per_round=(1 2 3 4 5)
rounds=(2)
epochs_per_round=(2)

# Open accuracy_file.txt and clear its contents
> accuracy_file.txt

# Loop through each combination of rounds and epochs per round
for r in "${rounds[@]}"; do
    for e in "${epochs_per_round[@]}"; do
        echo "Running fed_avg.py with $r rounds and $e epochs per round"
        python3.11 fed_avg.py --num_rounds $r --epochs $e  
    done
done

