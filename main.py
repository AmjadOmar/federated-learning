import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Simulate client data (e.g., 3 clients, each with their own data)
def create_client_data(num_clients, num_samples):
    clients_data = []
    for _ in range(num_clients):
        # Create random data (28x28 images flattened, and 10 classes)
        data = torch.randn(num_samples, 28 * 28)
        labels = torch.randint(0, 10, (num_samples,))
        dataset = TensorDataset(data, labels)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        clients_data.append(dataloader)
    return clients_data


# Local training on a client
def local_train(client_data, model, epochs=1, lr=0.01):
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data, labels in client_data:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


# Server function to aggregate client updates (averaging model parameters)
def aggregate_models(global_model, client_models):
    global_state_dict = global_model.state_dict()

    # Initialize the aggregation dictionary
    avg_state_dict = {key: torch.zeros_like(value) for key, value in global_state_dict.items()}

    # Sum the parameters from all clients
    for client_model in client_models:
        client_state_dict = client_model.state_dict()
        for key in avg_state_dict.keys():
            avg_state_dict[key] += client_state_dict[key]

    # Average the parameters
    for key in avg_state_dict.keys():
        avg_state_dict[key] /= len(client_models)

    # Load averaged weights into the global model
    global_model.load_state_dict(avg_state_dict)


# Simulate federated learning process
def federated_learning(num_clients=3, rounds=5):
    # Initialize global model
    global_model = SimpleNN()

    # Create local data for clients
    clients_data = create_client_data(num_clients, 1000)  # Each client has 1000 samples

    for r in range(rounds):
        print(f"Round {r + 1}")

        # Each client trains locally and sends the model back
        client_models = []
        for client_data in clients_data:
            # Create a copy of the global model
            local_model = SimpleNN()
            local_model.load_state_dict(global_model.state_dict())

            # Train locally
            local_train(client_data, local_model, epochs=1, lr=0.01)
            client_models.append(local_model)

        # Aggregate client models to update the global model
        aggregate_models(global_model, client_models)

    print("Federated learning process finished.")
    return global_model


# Run the federated learning simulation
global_model = federated_learning()
