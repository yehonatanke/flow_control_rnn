import torch
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from datasets import load_dataset
import torch.nn as nn

def load_sst2(train_size=1500, val_size=500):
    """Load SST-2 dataset and return splits."""
    dataset = load_dataset("glue", "sst2")
    train_set = dataset["train"][:train_size]
    val_set = dataset["validation"][:val_size]
    test_set = dataset["test"]
    return train_set, val_set, test_set

def build_vocab(sentences, min_freq=5, specials=["<unk>", "<pad>"]):
    """Build vocab from sentences."""
    vocab = build_vocab_from_iterator(map(str.split, sentences), specials=specials, min_freq=min_freq)
    vocab.set_default_index(vocab["<unk>"])
    return vocab

def tokenize_and_convert(sentences, vocab, max_len=50):
    """Tokenize and pad sentences."""
    tokenized = [torch.tensor([vocab[token] for token in sentence.split()]) for sentence in sentences]
    padded = torch.zeros(len(sentences), max_len, dtype=torch.long)
    lengths = torch.zeros(len(sentences), dtype=torch.long)
    for i, tokens in enumerate(tokenized):
        length = min(len(tokens), max_len)
        padded[i, :length] = tokens[:length]
        lengths[i] = length
    return padded, lengths

class SST2Dataset(Dataset):
    def __init__(self, data, vocab):
        self.sentences = [str(s).split() for s in data["sentence"]]
        self.labels = data["label"]
        self.vocab = vocab
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, idx):
        sentence = [self.vocab[token] for token in self.sentences[idx]]
        label = self.labels[idx]
        return torch.tensor(sentence, dtype=torch.long), torch.tensor(label, dtype=torch.long)

def collate_fn(batch):
    sentences, labels = zip(*batch)
    lengths = [len(s) for s in sentences]
    padded_sentences = nn.utils.rnn.pad_sequence(sentences, batch_first=True, padding_value=0)
    return padded_sentences, torch.tensor(labels), torch.tensor(lengths) 