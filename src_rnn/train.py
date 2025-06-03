import torch
import torch.nn as nn
import matplotlib.pyplot as plt

def train_and_plot(model, sentences, lengths, labels, seq_len, epochs=1):
    """Train RNNClassifier and plot gradient norms."""
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits = model(sentences, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    model.plot_grads(seq_len)

def train_and_evaluate(model, train_data, train_lengths, train_labels, val_data, val_lengths, val_labels, epochs=50, batch_size=32, plot_graph=True):
    """Train and evaluate model, plot loss/accuracy curves."""
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        correct_train, total_train = 0, 0
        for i in range(0, len(train_data), batch_size):
            batch_sentences = train_data[i:i+batch_size].to(device)
            batch_lengths = train_lengths[i:i+batch_size].to(device)
            batch_labels = train_labels[i:i+batch_size].to(device)
            optimizer.zero_grad()
            logits = model(batch_sentences, batch_lengths)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()
        avg_train_loss = total_train_loss / (len(train_data) // batch_size)
        train_losses.append(avg_train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)
        model.eval()
        total_val_loss = 0
        correct_val, total_val = 0, 0
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch_sentences = val_data[i:i+batch_size].to(device)
                batch_lengths = val_lengths[i:i+batch_size].to(device)
                batch_labels = val_labels[i:i+batch_size].to(device)
                logits = model(batch_sentences, batch_lengths)
                loss = criterion(logits, batch_labels)
                total_val_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total_val += batch_labels.size(0)
                correct_val += (predicted == batch_labels).sum().item()
        avg_val_loss = total_val_loss / (len(val_data) // batch_size)
        val_losses.append(avg_val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)
        print(f"[Epoch {epoch+1}/{epochs}] [Train Loss: {avg_train_loss:.4f}  Val Loss: {avg_val_loss:.4f}] [Train Acc: {100 * train_accuracy:.2f}%  Val Acc: {100 * val_accuracy:.2f}%]")
        if epoch > 10 and val_accuracies[-1] > 0.95 and train_losses[-1] < 0.1:
            print("Overfitting detected, stopping training.")
            break
    if plot_graph:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label="Train Accuracy")
        plt.plot(val_accuracies, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show() 