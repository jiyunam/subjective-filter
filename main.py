import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy
spacy_en = spacy.load('en')

import argparse

from models import Baseline, RNN, CNN

torch.manual_seed(100)
seed = 100


def evaluate(model, test_loader):
    tot_corr = 0
    tot_loss = 0
    loss_fnc = torch.nn.BCELoss()

    for i, t_batch in enumerate(test_loader):
        label = t_batch.label
        feats, lengths = t_batch.text

        predicts = model(feats, lengths)

        corr = (predicts > 0.5).squeeze().long() == label

        batch_loss = loss_fnc(input=predicts.squeeze(), target=label.float())
        tot_loss += batch_loss
        tot_corr += int(corr.sum())

    return float(tot_corr) / len(test_loader.dataset), tot_loss/(i+1)


def main(args):
    MaxEpochs = args.epochs
    lr = args.lr
    batchsize = args.batch_size
    curr_model = args.model

    # 3.2 Processing of the data
    TEXT = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
    LABEL = data.Field(sequential=False, use_vocab=False)

    train, val, test = data.TabularDataset.splits(path='/Users/jiyun/PycharmProjects/mie324/assign4', train='train.tsv',
                                                  validation='validation.tsv', test='test.tsv', format='tsv',
                                                  skip_header=True, fields=[('text', TEXT), ('label', LABEL)])

    train_iter, val_iter, test_iter = data.BucketIterator.splits(datasets=(train, val, test),
                                                                 sort_key=lambda x: len(x.text), sort_within_batch=True,
                                                                 repeat=False, batch_sizes=(batchsize,batchsize,batchsize),
                                                                 device=-1)
    # train_iter, val_iter, test_iter = data.Iterator.splits(datasets=(train, val, test),
    #                                                              sort_key=lambda x: len(x.text), sort_within_batch=True,
    #                                                              repeat=False,
    #                                                              batch_sizes=(batchsize, batchsize, batchsize),
    #                                                              device=-1)

    TEXT.build_vocab(train)
    vocab = TEXT.vocab
    vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))


    # 5 Training and Evaluation

    loss_fnc = torch.nn.BCELoss()

    base_model = Baseline(100, vocab)
    rnn_model = RNN(100, vocab, 100)
    cnn_model = CNN(100, vocab, 50, [2, 4])

    if curr_model == 'baseline':
        model = base_model
    elif curr_model == 'rnn':
        model = rnn_model
    elif curr_model == 'cnn':
        model = cnn_model

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(MaxEpochs):
        accum_loss = 0
        tot_corr = 0

        for i, batch in enumerate(train_iter):
            label = batch.label
            feats, lengths = batch.text

            optimizer.zero_grad()

            predicts = model(feats, lengths)

            batch_loss = loss_fnc(input=predicts.squeeze(), target=label.float())
            accum_loss += batch_loss
            batch_loss.backward()
            optimizer.step()

            corr = (predicts > 0.5).squeeze().long() == label
            tot_corr += int(corr.sum())

        train_acc = float(tot_corr)/(batchsize*100)
        train_loss = accum_loss/(batchsize*100)
        valid_acc, valid_loss = evaluate(model, val_iter)
        print("Epoch: {} | Train acc: {} | Train loss: {} | Valid acc: {} | Valid loss: {}".format(epoch, train_acc,
                                                                                                   train_loss, valid_acc,
                                                                                                   valid_loss))

    print('Finished Training')
    torch.save(model, "model_%s.pt" %(curr_model))
    test_model = torch.load("model_%s.pt" %(curr_model))
    test_acc, test_loss = evaluate(test_model, test_iter)
    # test_acc, test_loss = evaluate(model, test_iter)
    print('Test acc: {} | Test loss: {}'.format(test_acc, test_loss))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='baseline',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--emb-dim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--num-filt', type=int, default=50)

    args = parser.parse_args()

    main(args)

