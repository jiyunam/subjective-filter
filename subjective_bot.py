"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""

from models import Baseline, RNN, CNN
import torch
import torchtext
from torchtext import data
import spacy

base_model = torch.load('model_baseline.pt')
rnn_model = torch.load('model_rnn.pt')
cnn_model = torch.load('model_cnn.pt')

nlp = spacy.load('en')

TEXT = data.Field(sequential=True, include_lengths=True, tokenize='spacy')
LABEL = data.Field(sequential=False, use_vocab=False)

train, val, test = data.TabularDataset.splits(path='/Users/jiyun/PycharmProjects/mie324/assign4', train='train.tsv',
                                              validation='validation.tsv', test='test.tsv', format='tsv', skip_header=True,
                                              fields=[('text', TEXT), ('label', LABEL)])
TEXT.build_vocab(train)
vocab = TEXT.vocab
vocab.load_vectors(torchtext.vocab.GloVe(name='6B', dim=100))


while True:
    sentence = input('Enter a sentence \n')
    sentence = nlp(sentence)
    tokens = []
    length = []

    for token in sentence:
        tokens.append(vocab.stoi[token.text])

    tokens_tensor = torch.LongTensor(tokens)
    tokens_tensor = tokens_tensor.view(tokens_tensor.shape[0], 1)
    length.append(tokens_tensor.shape[0])
    length_tensor = torch.LongTensor(length)

    base_result = base_model(tokens_tensor, length)
    rnn_result = rnn_model(tokens_tensor, length)
    cnn_result = cnn_model(tokens_tensor, length)

    base_num = base_result[0][0].item()
    rnn_num = rnn_result[0][0].item()
    cnn_num = cnn_result[0].item()

    [base_res, rnn_res, cnn_res] = ['objective', 'objective', 'objective']
    if (base_num>0.5):
        base_res = 'subjective'
    if (rnn_num>0.5):
        rnn_res = 'subjective'
    if (cnn_num>0.5):
        cnn_res = 'subjective'

    print('Model baseline: %s (%s)'%(base_res, base_num))
    print('Model rnn: %s (%s)' % (rnn_res, rnn_num))
    print('Model cnn: %s (%s)' % (cnn_res, cnn_num))
    print('\n')




