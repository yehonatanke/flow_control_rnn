from data import load_sst2, build_vocab, tokenize_and_convert
from models import RNNClassifier, FlowControlRNN
from train import train_and_plot, train_and_evaluate
from utils import get_device
import torch

if __name__ == "__main__":
    # Load data
    train_set, val_set, test_set = load_sst2(train_size=500, val_size=100)
    vocab = build_vocab(train_set["sentence"], min_freq=5)
    train_sentences, train_lengths = tokenize_and_convert(train_set["sentence"], vocab)
    train_labels = torch.tensor(train_set["label"], dtype=torch.long)
    val_sentences, val_lengths = tokenize_and_convert(val_set["sentence"], vocab)
    val_labels = torch.tensor(val_set["label"], dtype=torch.long)
    device = get_device()
    
    # Train FlowControlRNN
    model = FlowControlRNN(vocab_size=len(vocab), embed_dim=50, hidden_dim=100, num_classes=2, pad_idx=vocab["<pad>"])
    model.to(device)
    train_and_evaluate(
        model,
        train_sentences,
        train_lengths,
        train_labels,
        val_sentences,
        val_lengths,
        val_labels,
        epochs=10,
        batch_size=32
    ) 