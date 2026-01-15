import matplotlib.pyplot as plt
import os

os.makedirs('results/plots', exist_ok=True)

def plot_metrics(train_losses, test_accuracies):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_losses, label='Train Loss')
    plt.xlabel('Round/Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Round/Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('results/plots/train_vs_val_accuracy.png')
    plt.close()
