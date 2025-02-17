import torch
import copy
import numpy as np
import csv
import time


from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel execution

from task import Net, load_data, train, test, get_weights, set_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")

NUM_CLIENTS = 10  # Adjust this if needed

def federated_avg(weights_list):
    """Compute federated averaging of model weights."""
    avg_weights = copy.deepcopy(weights_list[0])
    for key in avg_weights.keys():
        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key]
        avg_weights[key] = avg_weights[key] / len(weights_list)
    return avg_weights

def train_client(global_model, data,  client_id, epochs):
    """Train the local model on a specific client."""
    print(f"Training on client {client_id + 1}...")
    
    # Create a copy of the global model for this client
    local_model = copy.deepcopy(global_model)
    
    # Train the local model
    train_loss = train(local_model, data, epochs, DEVICE)
    
    return local_model.state_dict(), train_loss

def train_and_evaluate(num_rounds, epochs, data,  writer):
    """Simulate federated learning across multiple clients in parallel."""
    print(f"Training for {num_rounds} rounds for {epochs} epochs...")


    global_model = Net().to(DEVICE)

    for rnd in range(num_rounds):
        print(f"Round {rnd + 1}/{num_rounds}")

        local_weights = []
        local_losses = []

        # Train clients in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=NUM_CLIENTS) as executor:
            futures = [
                executor.submit(train_client, global_model, data[client_id], client_id, epochs)
                for client_id in range(NUM_CLIENTS)
            ]

            for future in as_completed(futures):
                weights, loss = future.result()
                local_weights.append(weights)
                local_losses.append(loss)

        # Federated averaging
        avg_weights = federated_avg(local_weights)
        global_model.load_state_dict(avg_weights)
        
        # Test the global model
        _, global_accuracy = test(global_model, load_data(0, NUM_CLIENTS, False)[1], DEVICE)
        avg_loss = np.mean(local_losses)
        
        print(f"Round {rnd + 1} results: avg loss = {avg_loss:.4f}, global accuracy = {global_accuracy:.4f}")

        # Write round results to CSV
        writer.writerow({"Number of Rounds": num_rounds, "Number of Epochs": epochs, "Round": rnd + 1,  "Global Accuracy": global_accuracy})

if __name__ == "__main__":
    num_rounds_array = [60]
    epochs_array = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


    start_time = time.time()

    data = []
    for client_id in range(NUM_CLIENTS): 
        current_data, _ = load_data(client_id, NUM_CLIENTS, False)
        data.append(current_data)

    with open("results/no_attack_results.csv", "w", newline="") as csvfile:
        fieldnames = ["Number of Rounds", "Number of Epochs", "Round", "Global Accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for num_rounds in num_rounds_array:
            for epochs in epochs_array:
                train_and_evaluate(num_rounds, epochs, data, writer)
                print("")
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    print("Federated training completed.")
