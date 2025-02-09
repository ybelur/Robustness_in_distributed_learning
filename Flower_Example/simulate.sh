#!/bin/bash

# Define arrays for rounds and epochs per round
# rounds=(70)
# epochs_per_round=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
rounds=(2)
epochs_per_round=(2)

# Open accuracy_file.txt and clear its contents
> accuracy_file.txt

# Loop through each combination of rounds and epochs per round
for r in "${rounds[@]}"; do
    for e in "${epochs_per_round[@]}"; do
        echo "Running flwr with $r rounds and $e epochs per round"
        echo "Running flwr with $r rounds and $e epochs per round" >> accuracy_file.txt
        flwr run fun/ --run-config "num-server-rounds=$r local-epochs=$e"
        echo "----------------------------------------" >> accuracy_file.txt
    done
done

