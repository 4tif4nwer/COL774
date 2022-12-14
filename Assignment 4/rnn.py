

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import sys
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# Can You see this update?


# import os
# for dirname, _, filenames in os.walk('data'):
#     for filename in filenames:
#         os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from sklearn.model_selection import train_test_split
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import nltk
import re

def clean_text(text):
    #2. remove unkonwn characrters
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
   
    #1. remove http links
    url = re.compile(r'https?://\S+|www\.\S+')
    text = url.sub(r'',text)
    
    #5. lowercase
    text = text.lower()
    
    return text

data = sys.argv[1]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
#print(os.listdir("data"))
train_x = pd.read_csv(data+"/train_x.csv")
train_y = pd.read_csv(data+"/train_y.csv")
test_x = pd.read_csv(data+"/non_comp_test_x.csv")
test_y = pd.read_csv(data+"/non_comp_test_y.csv")

train_x['Title'] = train_x['Title'].apply(lambda x: clean_text(x))
test_x['Title'] = test_x['Title'].apply(lambda x: clean_text(x))

from torchtext.vocab import GloVe, vocab
glove= GloVe(dim='300', name='6B')

train_y['Genre'][34199]

def get_words(df,df_y, glove_vector):
    train = []
    for i in range(len(df.index)):
        title = df['Title'][i]
        idxs = [glove_vector.stoi[w]        # lookup the index of word
                for w in title.split()
                if w in glove_vector.stoi] # keep words that has an embedding
        if not idxs: # ignore tweets without any word with an embedding
            continue
        idxs = torch.tensor(idxs) # convert list to pytorch tensor
        label = torch.tensor(int(df_y['Genre'][i])).long()
        train.append((idxs, label, i))
    return train
train = get_words(train_x,train_y,glove)
test = get_words(test_x,test_y,glove)

max_len = 0
for i in range(len(train)):
    max_len=max(max_len,len(train[i][0]))
print(max_len)

glove_emb = nn.Embedding.from_pretrained(glove.vectors)
print(train[4521][1])

rnn_layer = nn.RNN(input_size=300,    # dimension of the input repr
                   hidden_size=128,   # dimension of the hidden units
                   batch_first=True,
                   bidirectional=True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNN, self).__init__()
        self.emb = nn.Embedding.from_pretrained(glove.vectors)
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 128)
        self.ac1 = torch.nn.Tanh()
        self.fc2 = nn.Linear(128, num_classes)
    def forward(self, x):
        # Look up the embedding
        x = self.emb(x)
        # Forward propagate the RNN
        out, _ = self.rnn(x)
        # Pass the output of the last time step to the classifier
        out = self.fc1(out[:, -1, :])
        out = torch.nn.functional.tanh(out)
        out = self.fc2(out)
        
        return out

from torch.nn.utils.rnn import pad_sequence

title_padded = pad_sequence([title for title, label, indx in train[:32]],
                            batch_first=True)
title_padded.shape

import random

class Batcher:
    def __init__(self, titles, batch_size=32, drop_last=False):
        self.titles_by_length = {}
        for words, label, indx in titles:
            # compute the length of the tweet
            wlen = words.shape[0]
            # put the tweet in the correct key inside self.tweet_by_length
            if wlen not in self.titles_by_length:
                self.titles_by_length[wlen] = []
            self.titles_by_length[wlen].append((words, label,indx),)
         
        #  create a DataLoader for each set of titles of the same length
        self.loaders = {wlen : torch.utils.data.DataLoader(
                                    titles,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    drop_last=drop_last) # omit last batch if smaller than batch_size
            for wlen, titles in self.titles_by_length.items()}
    def __iter__(self): # called by Python to create an iterator
        # make an iterator for every tweet length
        iters = [iter(loader) for loader in self.loaders.values()]
        while iters:
            # pick an iterator (a length)
            im = random.choice(iters)
            try:
                yield next(im)
            except StopIteration:
                # no more elements in the iterator, remove it
                iters.remove(im)

def get_accuracy(model, data_loader):

    correct, total = 0, 0
    for titles, labels, indx in data_loader:
        titles = titles.to(device)
        labels = labels.to(device)
        output = model(titles)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(labels.view_as(pred)).sum().item()
        total += labels.shape[0]
    return correct / total

def train_rnn_network(model, train_loader, valid_loader, num_epochs=5, learning_rate=1e-5):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses, train_acc, valid_acc = [], [], []
    epochs = []
    for epoch in range(num_epochs):
        for titles, labels, indx in train_loader:
            titles = titles.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            pred = model(titles)
            loss = criterion(pred, labels)
            loss.backward()
            optimizer.step()
        losses.append(float(loss))

        epochs.append(epoch)
        train_acc.append(get_accuracy(model, train_loader))
        valid_acc.append(get_accuracy(model, valid_loader))
        print("Epoch %d; Loss %f; Train Acc %f; Val Acc %f" % (
              epoch+1, loss, train_acc[-1], valid_acc[-1]))
    # # plotting
    # plt.title("Training Curve")
    # plt.plot(losses, label="Train")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.show()

    # plt.title("Training Curve")
    # plt.plot(epochs, train_acc, label="Train")
    # plt.plot(epochs, valid_acc, label="Validation")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend(loc='best')
    # plt.show()

model = RNN(300, 128, 30)
model.to(device)
train_loader = Batcher(train, batch_size=32, drop_last=True)
valid_loader = Batcher(test, batch_size=32, drop_last=False)
train_rnn_network(model, train_loader, valid_loader, num_epochs=30, learning_rate=1e-4)
get_accuracy(model, valid_loader)

def get_words_test(df, glove_vector):
    train = []
    for i in range(len(df.index)):
        title = df['Title'][i]
        idxs = [glove_vector.stoi[w]        # lookup the index of word
                for w in title.split()
                if w in glove_vector.stoi] # keep words that has an embedding
        if not idxs: # ignore tweets without any word with an embedding
            continue
        idxs = torch.tensor(idxs) # convert list to pytorch tensor
        label = 0
        train.append((idxs, label,i))
    return train

ftest_x = pd.read_csv('data/non_comp_test_x.csv')
ftest_x['Title'] = ftest_x['Title'].apply(lambda x: clean_text(x))
ftest = get_words_test(ftest_x,glove)
ftest_loader = Batcher(ftest, batch_size=32, drop_last=False)

def get_accuracy_test(model, data_loader):
    a1=[]
    a2=[]
    for titles, labels, indx in data_loader:
        titles = titles.to(device)
        labels = labels.to(device)
        output = model(titles)
        pred = output.max(1, keepdim=True)[1]
        a1.append(pred)
        a2.append(indx)
    return a1,a2

a1,a2=get_accuracy_test(model,ftest_loader)
print(len(a2))

ans = torch.cat(a1)
indices = torch.cat(a2)
print(ans)

print(ans.shape)

ans=ans.to(torch.device("cpu"))
indices.to(torch.device("cpu"))
x_np_f = ans.numpy()
ind = indices.numpy()
sp_sort =np.argsort(ind)
x_np = x_np_f[sp_sort]
x_df = pd.DataFrame(x_np)
x_df.index.name='Id'
x_df.to_csv('non_comp_test_pred_y.csv',header=["Genre"])

