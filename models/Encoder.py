import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embedded_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedded_size)
        self.gru = nn.GRU(embedded_size, hidden_size)

    def forward(self, inputs, hidden=None):
        embedded = self.embedding(inputs)
        encoder_outputs, encoder_hidden = self.gru(embedded, hidden)
        return encoder_outputs, encoder_hidden

