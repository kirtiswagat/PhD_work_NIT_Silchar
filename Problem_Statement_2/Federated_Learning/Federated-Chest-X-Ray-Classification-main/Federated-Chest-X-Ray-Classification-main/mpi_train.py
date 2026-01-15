from mpi4py import MPI
import torch
import os
from federated.client import FederatedClient
from federated.server import FederatedServer
from federated.fed_avg import fed_avg
from utils.visualize import plot_metrics

COMM = MPI.COMM_WORLD
RANK = COMM.Get_rank()
SIZE = COMM.Get_size()

NUM_ROUNDS = 5
EPOCHS = 1

def main():
    # Ensure results directories
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)

    if RANK == 0:
        # Server process
        server = FederatedServer()
        global_model_params = None

        # For plotting
        round_accuracies = []
        round_sizes = []
        
        for round_idx in range(NUM_ROUNDS):
            print(f"Server: Starting round {round_idx+1}/{NUM_ROUNDS}")

            # Send global parameters to clients
            for client_rank in range(1, SIZE):
                COMM.send(global_model_params, dest=client_rank, tag=11)

            # Receive updated params and sizes from clients
            client_params_list = []
            client_sizes = []
            client_accuracies = []
            for client_rank in range(1, SIZE):
                client_data = COMM.recv(source=client_rank, tag=22)
                client_params_list.append(client_data['params'])
                client_sizes.append(client_data['size'])
                client_accuracies.append(client_data['accuracy'])

            # Aggregate using FedAvg
            global_model_params = fed_avg(client_params_list, client_sizes)
            print(f"Server: Completed aggregation for round {round_idx+1}")

            # Collect weighted test accuracy for this round
            total_clients = sum(client_sizes)
            weighted_acc = sum(acc * size for acc, size in zip(client_accuracies, client_sizes)) / total_clients
            round_accuracies.append(weighted_acc)
            round_sizes.append(total_clients)

            # Log round accuracy
            with open("results/logs/global_round_accuracy.txt", "a") as f:
                f.write(f"Round {round_idx+1}: Test Accuracy {weighted_acc:.4f}\n")

        print("Training Complete")

        # Save global model
        if global_model_params is not None:
            torch.save(global_model_params, "results/models/global_model.pt")
            print("Global model saved.")

        # Plot test accuracy curve
        plot_metrics([], round_accuracies)  # Empty list for train_loss if not available

    else:
        # Client process
        client_id = RANK
        client = FederatedClient(client_id)

        # Block waiting for initial params (None at first)
        global_params = COMM.recv(source=0, tag=11)
        if global_params:
            client.set_parameters(global_params)

        for round_idx in range(NUM_ROUNDS):
            print(f"Client {client_id}: Training round {round_idx+1}")

            if global_params:
                client.set_parameters(global_params)

            client.train(epochs=EPOCHS)
            acc = client.evaluate(round_idx + 1)  # Pass round_idx for log

            # Save client model after last round
            if round_idx == NUM_ROUNDS - 1:
                client.save_model()

            # Send updated params, sample size, and accuracy back to server
            params = client.get_parameters()
            data = {
                'params': params,
                'size': len(client.train_loader.dataset),
                'accuracy': acc
            }
            COMM.send(data, dest=0, tag=22)

            # Wait for updated global params
            global_params = COMM.recv(source=0, tag=11)

if __name__ == "__main__":
    main()