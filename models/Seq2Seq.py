import torch.nn as nn


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, inputs, targets):
        _, encoder_hidden = self.encoder.forward(inputs)
        decoder_outputs = self.decoder.forward(encoder_hidden, targets)
        return decoder_outputs







