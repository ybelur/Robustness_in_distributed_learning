"""fun: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters, NDArrays, Scalar
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from fun.task import Net, get_weights

import tensorflow as tf
from typing import Dict, Optional, Tuple

# def fit_metric(metrics):
#     train_loss = [num_examples * m["train_loss"] for num_examples, m in metrics]
#     examples = [num_examples for num_examples, _ in metrics]
#     print(metrics)
#     return {"train_loss": sum (train_loss) / sum (examples) }

def weighted_average (metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    accuracy = sum(accuracies) / sum(examples)
    with open("accuracy_file.txt", "a") as f:
        f.write(f"Accuracy: {accuracy}\n")

    return {"accuracy": sum (accuracies) / sum (examples) }

def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]

    # Initialize model parameters
    ndarrays = get_weights(Net())
    parameters = ndarrays_to_parameters(ndarrays)


    # Define strategy
    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)

