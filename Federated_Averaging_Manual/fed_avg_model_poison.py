import torch
import copy
import numpy as np
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel execution
from task import Net, load_data, train, test, get_weights, set_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def federated_avg(weights_list):
    """Compute federated averaging of model weights."""
    avg_weights = copy.deepcopy(weights_list[0])
    for key in avg_weights.keys():
        for i in range(1, len(weights_list)):
            avg_weights[key] += weights_list[i][key]
        avg_weights[key] = avg_weights[key] / len(weights_list)
    return avg_weights

def poison_model_weights(model_weights, scale_factor=1):
    """Introduce model poisoning by modifying the weights drastically."""
    poisoned_weights = copy.deepcopy(model_weights)
    for key in poisoned_weights.keys():
        poisoned_weights[key] = poisoned_weights[key] * scale_factor
    print("Model poisoning applied.")
    return poisoned_weights

def train_client(global_model, client_id, epochs, num_poisoned_clients, scale_factor):
    """Train the local model on a specific client, with optional model poisoning."""
    print(f"Training on client {client_id + 1}...")
    
    # Load client-specific data
    trainloader, _ = load_data(client_id, NUM_CLIENTS)
    
    # Create a copy of the global model for this client
    local_model = copy.deepcopy(global_model)
    
    # Train the local model
    train_loss = train(local_model, trainloader, epochs, DEVICE)
    
    # Apply model poisoning if the client_id is within the number of poisoned clients
    if client_id < num_poisoned_clients:
        poisoned_weights = poison_model_weights(local_model.state_dict(), scale_factor)
        return poisoned_weights, train_loss
    else:
        return local_model.state_dict(), train_loss

def train_and_evaluate(num_rounds, epochs, writer, num_poisoned_clients, scale_factor):
    """Simulate federated learning across multiple clients in parallel."""
    print(f"Training for {num_rounds} rounds for {epochs} epochs with {num_poisoned_clients} poisoned clients and scale factor {scale_factor}.")

    global NUM_CLIENTS
    NUM_CLIENTS = 5  # Adjust this if needed

    global_model = Net().to(DEVICE)

    for rnd in range(num_rounds):
        print(f"Round {rnd + 1}/{num_rounds}")

        local_weights = []
        local_losses = []

        # Train clients in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=NUM_CLIENTS) as executor:
            futures = [
                executor.submit(train_client, global_model, client_id, epochs, num_poisoned_clients, scale_factor)
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
        _, global_accuracy = test(global_model, load_data(0, NUM_CLIENTS)[1], DEVICE)
        avg_loss = np.mean(local_losses)
        
        print(f"Round {rnd + 1} results: avg loss = {avg_loss:.4f}, global accuracy = {global_accuracy:.4f}")

        # Write round results to CSV
        writer.writerow({"Number of Rounds": num_rounds, "Number of Epochs": epochs, "Scale Factor": scale_factor, "Round": rnd + 1, "Global Accuracy": global_accuracy})

if __name__ == "__main__":
    num_rounds_array = [50]
    epochs_array = [4]
    scale_factor_array = [1, 2, 3, 4, 5]
    num_poisoned_clients = 1

    start_time = time.time()

    with open("results/model_poison_results.csv", "w", newline="") as csvfile:
        fieldnames = ["Number of Rounds", "Number of Epochs", "Scale Factor",  "Round", "Global Accuracy"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for num_rounds in num_rounds_array:
            for epochs in epochs_array:
                for scale_factor in scale_factor_array:
                    train_and_evaluate(num_rounds, epochs, writer, num_poisoned_clients, scale_factor)
                    print("")
    
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    print("Federated training completed. Results saved in model_poison_results.csv.")
