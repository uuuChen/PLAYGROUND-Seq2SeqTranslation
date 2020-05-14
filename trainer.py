import torch.nn as nn
import numpy as np


class Trainer(object):
    def __init__(self, model, optimizer, args):
        self.model = model
        self.optimizer = optimizer
        self.criterion = None
        self.args = args

    def train(self, train_loader, val_loader):
        for epoch in range(self.args.epochs):
            self._train_epoch(train_loader, epoch)
            self._val_epoch(val_loader)

    def _train_epoch(self, train_loader, epoch):
        self.model.train()
        losses = AverageMeter()
        num_of_batches = train_loader.num_of_batches
        self.criterion = nn.CrossEntropyLoss(ignore_index=train_loader.PAD_IDX)
        for batch, (inputs_var, targets_var) in enumerate(train_loader.get_batch()):
            decoder_outputs = self.model(inputs_var, targets_var)
            self._show_pair_sequences(inputs_var, targets_var, decoder_outputs, train_loader, show_counts=5)
            loss = self.get_loss(decoder_outputs, targets_var)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss, count=len(inputs_var))
            print(f'epoch[{epoch}][{batch}/{num_of_batches}]\tloss: {losses.mean:.3f}')

    def _val_epoch(self, val_loader):
        self.model.eval()
        losses = AverageMeter()
        for batch, (inputs_var, targets_var) in enumerate(val_loader.get_batch()):
            decoder_outputs = self.model(inputs_var, targets_var)  # decoder_outputs: (ts, bs, output_size)
            self._show_pair_sequences(inputs_var, targets_var, decoder_outputs, val_loader, show_counts=5)
            loss = self.get_loss(decoder_outputs, targets_var)
            losses.update(loss, count=len(inputs_var))
        print(f'\n* validation loss: {losses.mean:.3f}\n')

    def get_loss(self, decoder_outputs, targets):
        time_steps, batch_size, output_size = decoder_outputs.shape
        decoder_outputs = decoder_outputs.reshape(time_steps * batch_size, output_size)
        targets = targets.reshape(-1)
        loss = self.criterion(decoder_outputs, targets)
        return loss

    def _show_pair_sequences(self, inputs, targets, outputs, data_loader, show_counts=1):
        inputs = inputs.detach().cpu().numpy().transpose(1, 0)  # (ts, bs) -> (bs, ts)
        targets = targets.detach().cpu().numpy().transpose(1, 0)  # (ts, bs) -> (bs, ts)
        outputs = outputs.detach().cpu().numpy().transpose(1, 0, 2)  # (ts, bs, output_size) -> (bs, ts, output_size)
        outputs_indices = np.argmax(outputs, axis=2)
        ori_src_sequences = data_loader.train_inputs_vocab.batch_indices_to_batch_sequences(inputs)
        ori_tar_sequences = data_loader.train_targets_vocab.batch_indices_to_batch_sequences(targets)
        pred_tar_sequences = data_loader.train_targets_vocab.batch_indices_to_batch_sequences(outputs_indices)
        triples_zip = zip(ori_src_sequences, ori_tar_sequences, pred_tar_sequences)
        for counts, (ori_src_seq, ori_tar_seq, pred_tar_seq) in enumerate(triples_zip, start=1):
            print('-' * 25)
            print(f"original source sequence: {' '.join(ori_src_seq)}")
            print(f"original target sequence: {' '.join(ori_tar_seq)}")
            print(f"predict  target sequence: {' '.join(pred_tar_seq)}")
            if counts == show_counts:
                break
        print('-' * 50)


class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.mean = 0

    def update(self, value, count=1):
        self.count += count
        self.sum += value * count
        self.mean = self.sum / self.count



