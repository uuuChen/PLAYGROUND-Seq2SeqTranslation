# system
import argparse
import torch
import os

# custom
from data_loader import DataLoader, Vocabulary
from models.Encoder import Encoder
from models.Decoder import Decoder
from models.Seq2Seq import Seq2Seq
from trainer import Trainer

###############################
# get parameters
###############################
parser = argparse.ArgumentParser()

# data paths
parser.add_argument("--train-input-path", type=str, default='data/train_en.txt')
parser.add_argument("--train-target-path", type=str, default='data/train_fr.txt')
parser.add_argument("--val-input-path", type=str, default='data/val_en.txt')
parser.add_argument("--val-target-path", type=str, default='data/val_fr.txt')
parser.add_argument("--test-input-path", type=str, default='data/test_en.txt')
# encoder
parser.add_argument("--encoder-embedded-size", type=int, default=256)
parser.add_argument("--encoder-hidden-size", type=int, default=256)
# decoder
parser.add_argument("--decoder-embedded-size", type=int, default=256)
parser.add_argument("--decoder-hidden-size", type=int, default=256)
parser.add_argument("--teacher-forcing-ratio", type=float, default=0.5)
# common
parser.add_argument("--no-use-cuda", action="store_true", default=False)
parser.add_argument("--load-model", "-lm", action="store_true", default=False)
parser.add_argument("--start-epoch", type=int, default=0)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--learning-rate", type=float, default=1e-3)
# save path
parser.add_argument("--save-dir-path", type=str, default='./saves/')

args = parser.parse_args()
use_cuda = not args.no_use_cuda and torch.cuda.is_available()
args.device = "cuda" if use_cuda else "cpu"


###############################
# get data loaders
###############################
train_inputs_vocab = Vocabulary(args.train_input_path)
train_targets_vocab = Vocabulary(args.train_target_path)
train_loader = DataLoader(train_inputs_vocab, train_targets_vocab, args.train_input_path, args.train_target_path, shuffle=True, batch_size=args.batch_size, device=args.device, is_train=True)
val_loader = DataLoader(train_inputs_vocab, train_targets_vocab, args.val_input_path, args.val_target_path, shuffle=False, batch_size=args.batch_size, device=args.device, is_train=True)
test_loader = DataLoader(train_inputs_vocab, train_targets_vocab, args.test_input_path, None, shuffle=False, batch_size=args.batch_size, device=args.device, is_train=False)


###############################
# get models
###############################
encoder = Encoder(train_loader.train_inputs_vocab.word_counts, args.encoder_embedded_size, args.encoder_hidden_size).to(args.device)
decoder = Decoder(train_loader.train_targets_vocab.word_counts, args.decoder_embedded_size, args.decoder_hidden_size, train_loader.SOS_IDX, train_loader.EOS_IDX, args.teacher_forcing_ratio, args.device).to(args.device)
seq2seq = Seq2Seq(encoder, decoder, args.device)


###############################
# get optimizer
###############################
optimizer = torch.optim.Adam(seq2seq.parameters(), lr=args.learning_rate)


###############################
# check direcotories exist
###############################
os.makedirs(args.save_dir_path, exist_ok=True)


def main():
    global seq2seq
    if args.load_model:
        seq2seq = torch.load(args.save_dir_path + 'model.pkl')
    trainer = Trainer(seq2seq, optimizer, args)
    trainer.train(train_loader, val_loader)
    torch.save(seq2seq, args.save_dir_path + 'model.pkl')
    trainer.predict(test_loader)


if __name__ == '__main__':
    main()
