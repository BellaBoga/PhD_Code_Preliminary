#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 10:10:46 2021

@author: bella
"""

import random
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import reconstruct_word
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from torchtext.data import Field, BucketIterator, TabularDataset
import torch
import spacy
import sys

def reconstruct_word(model, word1, max_length=1):
    tokens_ita = [token.lower() for token in list(word1) if type(word1)==str]
    tokens_ita.insert(0, ita.init_token)
    tokens_ita.append(ita.eos_token)
    text_to_indices_ita = [ita.vocab.stoi[token] for token in tokens_ita]    
    word_tensor_ita = torch.LongTensor(text_to_indices_ita).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        outputs_encoder1, hiddens, cells = model.encoder1(word_tensor_ita)

    outputs = [lat.vocab.stoi["<sos>"]]

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hiddens, cells = model.decoder(
                previous_word, outputs_encoder1, hiddens, cells)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the word
        if output.argmax(1).item() == lat.vocab.stoi["<eos>"]:
            break

    reconstruct_word = [lat.vocab.itos[idx] for idx in outputs]

    # remove start token
    return reconstruct_word[1:]


tokenize = lambda x: x.split(",")


ita = Field(sequential = True, tokenize = tokenize, use_vocab=True, init_token="<sos>", eos_token="<eos>")
lat = Field(sequential= True, tokenize = tokenize, use_vocab=True, init_token="<sos>", eos_token="<eos>")

fields = {"ita":('italian', ita), 'lat':("latin", lat)}

data = pd.read_csv("./data/rom-fra-ita-spa-por-lat.csv")

train_data, test_data = TabularDataset.splits(
        path = './data',
        train = 'train.csv',
        test= 'test.csv',
        #validation = "valid.csv",
        format = 'csv',
        fields =fields)

print(ita.build_vocab(train_data, max_size=8799, min_freq=2, specials = ['<pad>']))
lat.build_vocab(train_data, max_size=8799, min_freq=2, specials = ['<pad>'])

print(train_data[0].__dict__.keys())
print(train_data[1].__dict__.values())

class Encoder_Ita(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder_Ita, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size * 2, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (seq_length, N) where N is batch size

        embedding = self.dropout(self.embedding(x))
        print(embedding)
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states_ita, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states_ita, hidden, cell    
    
    
class Attention_Decoder(nn.Module):
    def __init__(
        self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(Attention_Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size * 2 + embedding_size, hidden_size, num_layers)

        self.energy = nn.Linear(hidden_size * 3, 1)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=0)
        self.relu = nn.ReLU()

    def forward(self, x, encoder_states_ita, hidden, cell):
        x = x.unsqueeze(0)
        # x: (1, N) where N is the batch size

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1, N, embedding_size)

        sequence_length = encoder_states_ita.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)
        # h_reshaped: (seq_length, N, hidden_size*2)

        energy = self.relu(self.energy(torch.cat((h_reshaped, encoder_states_ita), dim=2)))
        # energy: (seq_length, N, 1)

        attention = self.softmax(energy)
        # attention: (seq_length, N, 1)

        # attention: (seq_length, N, 1), snk
        # encoder_states: (seq_length, N, hidden_size*2), snl
        # we want context_vector: (1, N, hidden_size*2), i.e knl
        context_vector = torch.einsum("snk,snl->knl", attention, encoder_states_ita)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        # rnn_input: (1, N, hidden_size*2 + embedding_size)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        # outputs shape: (1, N, hidden_size)

        predictions = self.fc(outputs).squeeze(0)
        # predictions: (N, hidden_size)
        print(type(hidden))
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder1, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder1 = encoder1
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(lat_unique_sounds)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states_ita, hidden, cell = self.encoder1(source)

        # First input will be start of word token
        x = target[0]

        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            output, hidden, cell = self.decoder(x, encoder_states_ita, hidden, cell)

            # Store prediction for current time step
            try:
                outputs[t] = output
            except RuntimeError:
                pass

            # Get the best word the Decoder predicted 
            best_guess = output.argmax(1)

            # With probability of teacher_force_ratio we take the actual next word
            # otherwise we take the word that the Decoder predicted it to be.
            # Teacher Forcing is used so that the model gets used to seeing
            # similar inputs at training and testing time, if teacher forcing is 1
            # then inputs at test time might be completely different than what the
            # network is used to. This was a long comment.
            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs

#Training Hyperparameters
num_epochs = 100
#The learning rate is a hyperparameter that controls how much to change the model in response to 
#the estimated error each time the model weights are updated. 
learning_rate = 0.001
batch_size = 64

ita_unique_sounds = {'-', 'a', 'b', 'd', 'e', 'f', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'z', 'ŋ', 'ɔ', 'ɛ', 'ɡ', 'ɲ', 'ɾ', 'ʃ', 'ʎ', 'ʒ', 'ː'}

lat_unique_sounds = {'-', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'y', 'z', 'ŋ', 'ɔ', 'ɛ', 'ɡ', 'ɪ', 'ɹ', 'ʊ', 'ʰ', 'ʷ', 'ː'}
 
#model hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder1 = len(ita.vocab)
input_size_decoder = len(lat.vocab)
output_size = len(lat.vocab)
encoder_embedding_size = 100
decoder_embedding_size = 100
hidden_size = 1024
num_layers = 1 #if this is set to more layers I get a hidden[0] size error, apparently I set up sth weirdly
encoder_dropout = 0.5
decoder_dropout = 0.5


train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), 
                                                      batch_size=batch_size,
                                                      sort_within_batch = True,
                                                      sort_key = lambda x: len(x.italian),
                                                      device = device)
encoder_net1 = Encoder_Ita(input_size_encoder1, 
                      encoder_embedding_size, 
                      hidden_size, 
                      num_layers, 
                      encoder_dropout).to(device)
decoder_net = Attention_Decoder(input_size_decoder, 
                      decoder_embedding_size, 
                      hidden_size,
                      output_size,
                      num_layers, 
                      decoder_dropout).to(device)                     
                          
model = Seq2Seq(encoder_net1, decoder_net)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Tensorboard
writer = SummaryWriter(f"runs/loss_plot")
step = 0

pad_idx1 = ita.vocab.stoi["<pad>"]
criterion = nn.NLLLoss()

word1 = ("natsione")
word2 = ("naθjon")
word3 = ("nasjɔ̃")
word4 = ("nɐsɐ̃ʊ̃")
word5 = ("natsje")


for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    
    reconstructed_word = reconstruct_word(model, word1)
    
    with open('out.txt', 'a') as f:
        print(f"Reconstruct example words {word1, word2, word3, word4, word5,} \n as {reconstructed_word}", file=f)

    model.train()
    optimizer.zero_grad()
    model.eval()    
    
    for batch_idx, element in enumerate(train_iterator):
                #Get input and targets
                #batch_idx is the iterator and elememt are the elemelon in train_iterator
                inp_data = element.italian.to(device)
                targets = element.latin.to(device)
    
                #Forward propagation
                output = model(inp_data, targets)
                print("Output shape:", output.shape)
                print("Target shape:", targets.max(), targets.min())
                
    
                #Output is of shape (trg_len, batch_size, output_dim) but NLLLoss
                #doesn't take input in that form  so we need to do some reshaping
                output = output[1:].reshape(-1, output.shape[2])
                targets = targets[1:].reshape(-1)
                print("Output shape 2:", output.shape)
                print("Target shape 2:", targets.max(), targets.min())
                
                #In PyTorch, we need to set the gradients to zero before starting to do backpropragation because                  
                #PyTorch accumulates the gradients on subsequent backward passes. 
                #This is convenient while training RNNs. 
                #So, the default action is to accumulate (i.e. sum) the gradients on every loss.backward() call.
                
                #zero_grad() restarts looping without losses from the last step 
                #if you use the gradient method for decreasing the error (or losses).
                #If you do not use zero_grad() the loss will increase not decrease as required.
                
                optimizer.zero_grad()
                loss = criterion(output, targets)
                print("Loss:", loss)

                #Back propagation
                loss.backward()
                

                #Clip in order to avoid exploding gradient issues 
                #makes sure gradients are in reasonable range
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    
                #Gradient descent step
                optimizer.step()
             
                #Calculate training accuracy
                _, predictions = output.max(1)
                print("Preds:", predictions)
                num_correct = (predictions == targets).sum()
                running_training_accuracy = float(num_correct)//float(inp_data[0].item())
             
             
                # Plot to tensorboard
                writer.add_scalar("Training loss", loss, global_step=step)
                writer.add_scalar("Training Accuracy", running_training_accuracy, global_step=step)
                step += 1
                

#Questions Maxim: nice way to evaluat? loss function, what's going on? 
                
                
# <class 'torch.Tensor'>
# <class 'torch.Tensor'>
# Output shape: torch.Size([3, 64, 32])
# Target shape: tensor(3) tensor(0)
# Output shape 2: torch.Size([128, 32])
# Target shape 2: tensor(3) tensor(0)
# tensor(3.4657) --> Konstante ; no gradients for them = Maybe the issue?