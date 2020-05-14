import torch.nn as nn
import torch
import random
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, output_size, embedded_size, hidden_size, sos_idx, teacher_forcing_ratio, device):
        super(Decoder, self).__init__()
        self.sos_idx = sos_idx
        self.output_size = output_size
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.device = device

        self.embedding = nn.Embedding(output_size, embedded_size)
        self.gru = nn.GRU(embedded_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def _forward_step(self, input, hidden):
        embedded = torch.unsqueeze(self.embedding(input), dim=0)  # (bs, embedded_size) -> (1, bs, embedded_size)
        rnn_output, rnn_hidden = self.gru(embedded, hidden)
        output = self.out(torch.squeeze(rnn_output, dim=0))  # # output_t: (1, bs, output_size) -> (bs, output_size)
        return output, rnn_hidden

    def forward(self, encode_vec, targets):
        batch_size = encode_vec.shape[1]
        max_target_len = targets.shape[0]
        decoder_hidden = encode_vec
        decoder_input = Variable(torch.LongTensor([self.sos_idx] * batch_size)).to(self.device)
        decoder_outputs = Variable(torch.zeros(max_target_len, batch_size, self.output_size)).to(self.device)

        use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False

        for t in range(max_target_len):
            output, decoder_hidden = self._forward_step(decoder_input, decoder_hidden)  # output: (bs, output_size)
            decoder_outputs[t] = output
            if use_teacher_forcing:
                decoder_input = targets[t, :]  # (bs,)
            else:
                decoder_input = torch.argmax(output, dim=1)  # (bs,)

        return decoder_outputs







