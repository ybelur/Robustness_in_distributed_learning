import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import csv
import time
import argparse

from concurrent.futures import ThreadPoolExecutor, as_completed 
from task import Net, load_data, train, test, get_weights, set_weights

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on {DEVICE}")


def federated_avg(weights_list, aggregation_type, poison_probabilities, trim_ratio):
    """Compute federated averaging of model weights."""

    if aggregation_type == "mean":
        avg_weights = copy.deepcopy(weights_list[0])
        for key in avg_weights.keys():
            for i in range(1, len(weights_list)):
                avg_weights[key] += weights_list[i][key]  
            avg_weights[key] = avg_weights[key] / len(weights_list)
        return avg_weights
    
    elif aggregation_type == "median":
        median_weights = copy.deepcopy(weights_list[0])
        for key in median_weights.keys():
            stacked_weights = torch.stack([w[key] for w in weights_list])

            # print(f"Stacked weights for key {key}: {stacked_weights}")
            median_weights[key] = torch.median(stacked_weights, dim=0).values
            # print(f"Median weights for key {key}: {median_weights[key]}")

        return median_weights

    elif aggregation_type == "trimmed_mean":
        trimmed_mean_weights = copy.deepcopy(weights_list[0])
        # trim_ratio = 0.2  # Define the trim ratio (10% trimming)
        for key in trimmed_mean_weights.keys():
            stacked_weights = torch.stack([w[key] for w in weights_list])
            sorted_weights, _ = torch.sort(stacked_weights, dim=0)
            trim_count = int(trim_ratio * len(weights_list))
            trimmed_weights = sorted_weights[trim_count:-trim_count]  # Trim the extremes
            trimmed_mean_weights[key] = torch.mean(trimmed_weights, dim=0)
        return trimmed_mean_weights
    
    elif aggregation_type == "weighted_mean":
        probability_weights = [1 - p for p in poison_probabilities]
        total_weight = sum(probability_weights)

        avg_weights = copy.deepcopy(weights_list[0])

        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key])
            for i in range(len(weights_list)):
                avg_weights[key] += weights_list[i][key] * (probability_weights[i])
            avg_weights[key] = avg_weights[key] / total_weight
        return avg_weights
    
    elif aggregation_type == "threshold_exclude_mean":
        avg_weights = copy.deepcopy(weights_list[0])
        valid_weights = [weights_list[i] for i in range(len(weights_list)) if poison_probabilities[i] <= 0.5]

        print(f"Initial Weights: {len(weights_list)}")
        print(f"Valid Weights: {len(valid_weights)}")
        
        if not valid_weights:
            # If all clients are considered poisoned, return the first client's weights
            return weights_list[0]
        
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key])
            for weights in valid_weights:
                avg_weights[key] += weights[key]
                
            avg_weights[key] = avg_weights[key] / len(valid_weights)
        
        return avg_weights
    
    elif aggregation_type == "threshold_exclude_median":
        median_weights = copy.deepcopy(weights_list[0])
        valid_weights = [weights_list[i] for i in range(len(weights_list)) if poison_probabilities[i] <= 0.5]

        if not valid_weights:
            return weights_list[0]

        for key in median_weights.keys():
            stacked_weights = torch.stack([w[key] for w in valid_weights])
            median_weights[key] = torch.median(stacked_weights, dim=0).values
        return median_weights
    
    
    elif aggregation_type == "softmax_mean":
        # Use softmax over (1 - poison_probabilities) as weights
        prob_scores = np.array([1 - p for p in poison_probabilities])
        # Softmax for normalized weighting
        exp_scores = np.exp(prob_scores)
        softmax_weights = exp_scores / np.sum(exp_scores)
        print(f"Softmax Weights: {softmax_weights}")
        print(f'Softmax Sum: {np.sum(softmax_weights)}')
        avg_weights = copy.deepcopy(weights_list[0])
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key])
            for i in range(len(weights_list)):
                avg_weights[key] += weights_list[i][key] * softmax_weights[i]
        return avg_weights
    
    elif aggregation_type == "dropout_mean":

        avg_weights = copy.deepcopy(weights_list[0])
        random_probs = np.random.rand(len(weights_list))
        valid_weights = [weights_list[i] for i in range(len(weights_list)) if random_probs[i] > poison_probabilities[i]]

        # print("Random Probabilities: ", random_probs)
        # print("Poison Probabilities: ", poison_probabilities)
        # print(f"Valid Weights Indices: ", [i for i in range(len(weights_list)) if random_probs[i] > poison_probabilities[i]])

        if not valid_weights:
            return weights_list[np.random.randint(len(weights_list))]
        for key in avg_weights.keys():
            avg_weights[key] = torch.zeros_like(avg_weights[key])
            for weights in valid_weights:
                avg_weights[key] += weights[key]
                
            avg_weights[key] = avg_weights[key] / len(valid_weights)
        return avg_weights
    
    elif aggregation_type == "dropout_median":
        
        median_weights = copy.deepcopy(weights_list[0])
        random_probs = np.random.rand(len(weights_list))
        valid_weights = [weights_list[i] for i in range(len(weights_list)) if random_probs[i] > poison_probabilities[i]]

        if not valid_weights:
            return weights_list[np.random.randint(len(weights_list))]

        for key in median_weights.keys():
            stacked_weights = torch.stack([w[key] for w in valid_weights])
            median_weights[key] = torch.median(stacked_weights, dim=0).values
        return median_weights
    
    elif aggregation_type == "weighted_krum":
        num_clients = len(weights_list)
        vectors = [torch.cat([v.flatten() for v in w.values()]) for w in weights_list]
        vectors = torch.stack(vectors)
        scores = []
        for i in range(num_clients):
            dists = [torch.norm(vectors[i] - vectors[j])**2 for j in range(num_clients) if j != i]
            dists = sorted(dists)[:num_clients - 2]
            weighted_score = sum((1 - poison_probabilities[j]) * d for j, d in enumerate(dists))
            scores.append(weighted_score)
        selected = np.argmin(scores)
        return copy.deepcopy(weights_list[selected])
    
    else:
        raise ValueError("Unsupported aggregation type.")


def poison_model_weights(model_weights, scale_factor):
    """Introduce model poisoning by modifying the weights drastically."""
    
    poisoned_weights = copy.deepcopy(model_weights)
    for key in poisoned_weights.keys():
        poisoned_weights[key] = poisoned_weights[key] * scale_factor
    print("Model poisoning applied.")
    return poisoned_weights

def train_client(global_model, data, client_id, epochs, num_model_poisoned_clients, scale_factor):
    """Train the local model on a specific client, with optional data poisoning."""

    print(f"Training on client {client_id + 1}...")
    
    # Create a copy of the global model for this client
    local_model = copy.deepcopy(global_model)
    
    # Train the local model
    train_loss = train(local_model, data, epochs, DEVICE)
    
    # Apply model poisoning if the client_id is within the number of poisoned clients
    if client_id < num_model_poisoned_clients:
        poisoned_weights = poison_model_weights(local_model.state_dict(), scale_factor)
        return poisoned_weights, train_loss
    else:
        return local_model.state_dict(), train_loss


def train_and_evaluate(num_clients, num_rounds, epochs, data, writer, num_data_poisoned_clients, num_model_poisoned_clients, scale_factor, aggregation_type, poison_probabilities, trim_ratio):
    """Simulate federated learning across multiple clients in parallel."""

    print("Now Training with these parameters:")
    print(f"Number of Clients: {num_clients}, Number of Rounds: {num_rounds}, Number of Epochs: {epochs}, Number of Data Poisoned Clients: {num_data_poisoned_clients}, Number of Model Poisoned Clients: {num_model_poisoned_clients}, Scale Factor: {scale_factor}, Aggregation Type: {aggregation_type}")

    global_model = Net().to(DEVICE)

    for rnd in range(num_rounds):
        print(f"Round {rnd + 1}/{num_rounds}")

        local_weights = []
        local_losses = []

        # Train clients in parallel using ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=num_clients) as executor:
            futures = [
                executor.submit(train_client, global_model, data[client_id], client_id, epochs, num_model_poisoned_clients, scale_factor)
                for client_id in range(num_clients)
            ]

            for future in as_completed(futures):
                weights, loss = future.result()
                local_weights.append(weights)
                local_losses.append(loss)

        # Federated averaging
        avg_weights = federated_avg(local_weights, aggregation_type, poison_probabilities, trim_ratio)
        global_model.load_state_dict(avg_weights)
        
        # Test the global model
        _, global_accuracy = test(global_model, load_data(0, num_clients, False)[1], DEVICE)
        avg_loss = np.mean(local_losses)
        
        print(f"Round {rnd + 1} results: avg loss = {avg_loss:.4f}, global accuracy = {global_accuracy:.4f}")

        # Write round results to CSV        
        writer.writerow({"Number of Clients": num_clients, "Number of Rounds": num_rounds, 
                         "Number of Epochs": epochs, "Number of Data Poisoned Clients": num_data_poisoned_clients, 
                         "Number of Model Poisoned Clients": num_model_poisoned_clients, 
                         "Scale Factor": scale_factor, "Aggregation Type": aggregation_type, 
                         "Round": rnd + 1, "Global Accuracy": global_accuracy})
        


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run federated training with data poisoning.")
    
    parser.add_argument("--num_clients", type=str, required=True, help="Number of clients.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the output CSV file.")
    parser.add_argument("--num_rounds", type=str, required=True, help="Comma-separated list of num_rounds values.")
    parser.add_argument("--epochs", type=str, required=True, help="Comma-separated list of epochs values.")
    parser.add_argument("--num_data_poisoned_clients", type=str, required=True, help="Comma-separated list of num_data_poisoned_clients values.")
    parser.add_argument("--scale_factor", type=str, required=True, help="Comma-separated list of scale_factor values.")
    parser.add_argument("--num_model_poisoned_clients", type=str, required=True, help="Comma-separated list of num_model_poisoned_clients values.")
    parser.add_argument("--aggregation_type", type=str, required=True, help="Aggregation type for federated averaging.")
    parser.add_argument("--trim_ratio", type=float, default=0.1, help="Trim ratio for trimmed mean aggregation (default: 0.1).")

    return parser.parse_args()

if __name__ == "__main__":

    args = parse_arguments()

    num_clients = int(args.num_clients)
    csv_file = args.csv_file
    aggregation_type = args.aggregation_type
    trim_ratio = args.trim_ratio

    # Convert comma-separated string arguments to lists of integers
    num_rounds_array = [int(x) for x in args.num_rounds.split(",")]
    epochs_array = [int(x) for x in args.epochs.split(",")]
    num_data_poisoned_clients_array = [int(x) for x in args.num_data_poisoned_clients.split(",")]
    scale_factor_array = [int(x) for x in args.scale_factor.split(",")]
    num_model_poisoned_clients_array = [int(x) for x in args.num_model_poisoned_clients.split(",")]

    start_time = time.time()

    # Print all parameters
    print("Parameters:")
    print(f"Number of Clients: {num_clients}")
    print(f"CSV File: {csv_file}")
    print(f"Number of Rounds: {num_rounds_array}")
    print(f"Epochs: {epochs_array}")
    print(f"Number of Data Poisoned Clients: {num_data_poisoned_clients_array}")
    print(f"Number of Model Poisoned Clients: {num_model_poisoned_clients_array}")
    print(f"Scale Factor: {scale_factor_array}")
    print(f"Aggregation Type: {aggregation_type}")
    print(f"Trim Ratio: {trim_ratio}")
    print("\n")

    with open(csv_file, "w", newline="") as csvfile:
        fieldnames = ["Number of Clients", "Number of Rounds",
                      "Number of Epochs", "Number of Data Poisoned Clients",
                      "Number of Model Poisoned Clients",
                      "Scale Factor", "Aggregation Type", 
                      "Round", "Global Accuracy"]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for num_data_poisoned_clients in num_data_poisoned_clients_array:
            # Load data for all clients
            data = []

            # Assign probabilities for each client being data poisoned
            probabilities = np.random.rand(num_clients)
            # probabilities = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            rand_array = np.random.rand(num_clients)
            
            # Based on the probabilities, determine which clients are data poisoned
            is_data_poisoned = rand_array < probabilities
            

            for client_id in range(num_clients):
                
                print(f"Random Array: {rand_array[client_id]}")
                print(f"Client {client_id + 1} probabilities: {probabilities[client_id]}")
                print(f"Client {client_id + 1} data poisoned: {is_data_poisoned[client_id]}")

                # current_data, _ = load_data(client_id, num_clients, (client_id < num_data_poisoned_clients))
                current_data, _ = load_data(client_id, num_clients, is_data_poisoned[client_id])
                data.append(current_data)

            for num_model_poisoned_clients in num_model_poisoned_clients_array:
                for scale_factor in scale_factor_array:
                    for num_rounds in num_rounds_array:
                        for epochs in epochs_array:
                            train_and_evaluate( num_clients=num_clients, 
                                                num_rounds=num_rounds, 
                                                epochs=epochs, 
                                                data=data, 
                                                writer=writer, 
                                                num_data_poisoned_clients=num_data_poisoned_clients, 
                                                num_model_poisoned_clients=num_model_poisoned_clients, 
                                                scale_factor=scale_factor,
                                                aggregation_type=aggregation_type,
                                                poison_probabilities=probabilities,
                                                trim_ratio=trim_ratio)

    csvfile.close()

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Start time: {time.ctime(start_time)}, End time: {time.ctime(end_time)}")
    print(f"Elapsed time: {time.strftime('%d days, %H Hours %M Minutes %S Seconds', time.gmtime(elapsed_time))}")
    print(f"Federated training completed. Results saved in {csv_file}")
