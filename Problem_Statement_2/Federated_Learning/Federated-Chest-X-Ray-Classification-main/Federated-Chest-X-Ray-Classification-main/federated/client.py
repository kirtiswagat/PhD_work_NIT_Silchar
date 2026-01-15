import os
import torch
import torch.nn as nn
import torch.optim as optim
from models.cnn_model import SimpleCNN
from data.data_loader import get_client_loader
from utils.metrics import accuracy

class FederatedClient:
    def __init__(self, client_id, device='cpu', lr=0.001):
        self.client_id = client_id
        self.device = device
        self.model = SimpleCNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.train_loader = get_client_loader(client_id, "train")
        self.test_loader = get_client_loader(client_id, "test")
        os.makedirs('results/logs', exist_ok=True)
        os.makedirs('results/models', exist_ok=True)

    def train(self, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Client {self.client_id} Epoch {epoch+1} Loss: {total_loss/len(self.train_loader):.4f}")

    def evaluate(self, round_idx=None):
        self.model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        print(f"Client {self.client_id} Test Accuracy: {acc:.4f}")
        # Save accuracy log
        log_path = f'results/logs/client_{self.client_id}_log.txt'
        with open(log_path, 'a') as f:
            if round_idx is not None:
                f.write(f"Round {round_idx}: {acc:.4f}\n")
            else:
                f.write(f"Test Accuracy: {acc:.4f}\n")
        return acc

    def save_model(self, round_idx=None):
        model_path = f'results/models/client_{self.client_id}_model.pt'
        torch.save(self.model.state_dict(), model_path)
        # If desired, version by round: f'results/models/client_{self.client_id}_model_round{round_idx}.pt'

    def get_parameters(self):
        return {k: v.cpu() for k, v in self.model.state_dict().items()}

    def set_parameters(self, parameters):
        self.model.load_state_dict(parameters)