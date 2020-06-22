import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()
        # 4.1 YOUR CODE HERE
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc1 = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        # 4.1 YOUR CODE HERE
        x = self.embedding(x)
        x = x.mean(0)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x


class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()
        # 4.2 YOUR CODE HERE
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.GRU = nn.GRU(embedding_dim, hidden_dim)
        self.fc1 = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        # 4.2 YOUR CODE HERE
        x = self.embedding(x)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths)
        output, hidden = self.GRU(x)
        x = self.fc1(hidden)
        x = torch.sigmoid(x)
        return x


class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes):
        super(CNN, self).__init__()

        # 4.3 YOUR CODE HERE
        self.vocab = vocab
        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)

        self.conv1 = nn.Conv1d(embedding_dim, n_filters, filter_sizes[0])
        self.conv2 = nn.Conv1d(embedding_dim, n_filters, filter_sizes[1])
        self.fc1 = nn.Linear(embedding_dim, 1)

    def forward(self, x, lengths=None):
        # 4.3 YOUR CODE HERE
        x = self.embedding(x)
        x1 = self.conv1(x.permute(1, 2, 0))
        x2 = self.conv2(x.permute(1, 2, 0))
        maxpool1 = nn.MaxPool1d(x1.shape[2])
        maxpool2 = nn.MaxPool1d(x2.shape[2])
        x1 = maxpool1(x1)
        x2 = maxpool2(x2)
        x = torch.cat([x1, x2], 1).squeeze()
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

