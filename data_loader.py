import random
import torch
from torch.autograd import Variable


class Vocabulary:
    def __init__(self, text_file_path):
        # initialize parameter
        self.word2idx = dict()
        self.sequences = list()
        self.indices = list()  # corresponding indices of sequences
        self.max_length = 0
        self.word_counts = 0

        # build vocabulary
        self._build(text_file_path)

    def _build(self, text_file_path):
        signals = ['SOS', 'EOS', 'PAD', 'UNK']
        with open(text_file_path, "r", encoding="utf-8-sig") as fp:
            lines = fp.readlines()
            for line in lines:
                sequence = line.strip().split(' ')
                self.sequences.append(sequence)
                if len(sequence) > self.max_length:
                    self.max_length = len(sequence)
        flat_sequences = [word for sequence in self.sequences for word in sequence]
        unique_words = signals + list(set(flat_sequences))
        self.word_counts = len(unique_words)
        self.word2idx = dict(zip(unique_words, range(len(unique_words))))
        self.idx2word = dict(zip(range(len(unique_words)), unique_words))
        self.indices = [self.sequence_to_indices(sequence, add_eos=True) for sequence in self.sequences]

    def sequence_to_indices(self, sequence, add_sos=False, add_eos=False):
        indices = list()
        if add_sos:
            indices.append(self.word2idx['SOS'])
        for word in sequence:
            index = self.word2idx['UNK'] if word not in self.word2idx else self.word2idx[word]
            indices.append(index)
        if add_eos:
            indices.append(self.word2idx['EOS'])
        return indices

    def batch_indices_to_batch_sequences(self, batch_indices):
        batch_sequences = list()
        for indices in batch_indices:
            sequence = list()
            for index in indices:
                word = self.idx2word[index]
                if word == 'EOS':
                    break
                sequence.append(word)
            batch_sequences.append(sequence)
        return batch_sequences


class DataLoader:
    def __init__(self, inputs_file_path, target_file_path, device=None, batch_size=1, shuffle=False):
        self.device = device
        self.num_of_batches = None
        self.batch_size = batch_size

        self.inputs_vocab = Vocabulary(inputs_file_path)
        self.target_vocab = Vocabulary(target_file_path)

        self.inputs = self.inputs_vocab.indices
        self.targets = self.target_vocab.indices

        self.PAD_IDX = self.inputs_vocab.word2idx['PAD']

        if shuffle:
            inputs_targets_list = list(zip(self.inputs, self.targets))
            random.shuffle(inputs_targets_list)
            self.inputs, self.targets = zip(*inputs_targets_list)

        self.inputs_lens = [len(input) for input in self.inputs]
        self.targets_lens = [len(target) for target in self.targets]

        self.batches = [
            ((self.inputs[k: k + self.batch_size], max(self.inputs_lens[k: k + self.batch_size])),
             (self.targets[k: k + self.batch_size], max(self.targets_lens[k: k + self.batch_size])))
            for k in range(0, len(self.inputs), self.batch_size)
        ]
        self.num_of_batches = len(self.batches)

    def get_batch(self):
        for batch in self.batches:
            (inputs, max_input_len), (targets, max_target_len) = batch

            padded_inputs = self._pad_sequences(inputs, max_input_len)
            padded_targets = self._pad_sequences(targets, max_target_len)

            inputs_var = Variable(torch.LongTensor(padded_inputs)).transpose(0, 1).to(self.device)  # time * batch
            targets_var = Variable(torch.LongTensor(padded_targets)).transpose(0, 1).to(self.device)  # time * batch

            yield inputs_var, targets_var

    def _pad_sequences(self, sequences, max_length):
        pad_sequences = list()
        for sequence in sequences:
            pad_sequence = sequence + [self.PAD_IDX] * (max_length - len(sequence))
            pad_sequences.append(pad_sequence)
        return pad_sequences


if __name__ == '__main__':
    train_loader = DataLoader('data/train_en.txt', 'data/train_fr.txt', shuffle=True, batch_size=128, device='cuda')
    for inputs, label in train_loader.get_batch():
        print(inputs.shape, label.shape)
