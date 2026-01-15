import torch

def accuracy(output, target):
    """Calculate classification accuracy."""
    with torch.no_grad():
        preds = torch.argmax(output, dim=1)
        correct = (preds == target).sum().item()
        return correct / target.size(0)