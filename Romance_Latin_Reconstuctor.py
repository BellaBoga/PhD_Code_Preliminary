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


def reconstruct_word(model, word1, word2, word3, word4, word5, device, max_length=1):
    tokens_spa = [token.lower() for token in list(word2) if type(word2)==str]
    tokens_por = [token.lower() for token in list(word5) if type(word5)==str]
    tokens_rom = [token.lower() for token in list(word4) if type(word4)==str]
    tokens_fra = [token.lower() for token in list(word3) if type(word3)==str]
    tokens_ita = [token.lower() for token in list(word1) if type(word1)==str]
    print("Tokens ita:", tokens_spa, len(tokens_spa))


    # Add <SOS> and <EOS> in beginning and end respectively
    tokens_spa.insert(0, spa.init_token)
    tokens_por.insert(0, por.init_token)
    tokens_rom.insert(0, rom.init_token)
    tokens_fra.insert(0, fra.init_token)
    tokens_ita.insert(0, ita.init_token)
    
    tokens_spa.append(spa.eos_token)
    tokens_por.append(por.eos_token)
    tokens_rom.append(rom.eos_token)
    tokens_fra.append(fra.eos_token)
    tokens_ita.append(ita.eos_token)
    print("Tokens spa with sos and eos:", tokens_spa)

    

    # Go through each ita token and convert to an index
    
    text_to_indices_spa = [spa.vocab.stoi[token] for token in tokens_spa]
    text_to_indices_por = [por.vocab.stoi[token] for token in tokens_por]
    text_to_indices_rom = [rom.vocab.stoi[token] for token in tokens_rom]
    text_to_indices_fra = [fra.vocab.stoi[token] for token in tokens_fra]
    text_to_indices_ita = [ita.vocab.stoi[token] for token in tokens_ita]
    

    # Convert to Tensor
    
    word_tensor_spa = torch.LongTensor(text_to_indices_spa).unsqueeze(1).to(device)
    word_tensor_por = torch.LongTensor(text_to_indices_por).unsqueeze(1).to(device)
    word_tensor_rom = torch.LongTensor(text_to_indices_rom).unsqueeze(1).to(device)
    word_tensor_fra = torch.LongTensor(text_to_indices_fra).unsqueeze(1).to(device)
    word_tensor_ita = torch.LongTensor(text_to_indices_ita).unsqueeze(1).to(device)

    # Build encoder hidden, cell state
    with torch.no_grad():
        outputs_encoder1, hiddens, cells = model.encoder1(word_tensor_spa)
        outputs_encoder2, hiddens, cells = model.encoder2(word_tensor_por)
        outputs_encoder3, hiddens, cells = model.encoder3(word_tensor_rom)
        outputs_encoder4, hiddens, cells = model.encoder4(word_tensor_fra)
        outputs_encoder5, hiddens, cells = model.encoder5(word_tensor_ita)

    outputs = [lat.vocab.stoi["<sos>"]]
    print("Outputs: ", outputs)

    for _ in range(max_length):
        previous_word = torch.LongTensor([outputs[-1]]).to(device)

        with torch.no_grad():
            output, hiddens, cells = model.decoder(
                previous_word, outputs_encoder1, outputs_encoder2, outputs_encoder3,                      
                outputs_encoder4, outputs_encoder5, hiddens, cells)
            best_guess = output.argmax(1).item()

        outputs.append(best_guess)

        # Model predicts it's the end of the word
        if output.argmax(1).item() == lat.vocab.stoi["<eos>"]:
            break

    reconstruct_word = [lat.vocab.itos[idx] for idx in outputs]

    # remove start token
    return reconstruct_word[1:]


tokenize = lambda x: x.split(",")

spa = Field(sequential = True, tokenize = tokenize, use_vocab=True, init_token="<sos>", eos_token="<eos>")
por = Field(sequential = True, tokenize = tokenize, use_vocab=True, init_token="<sos>", eos_token="<eos>")
rom = Field(sequential = True, tokenize = tokenize, use_vocab=True, init_token="<sos>", eos_token="<eos>")
fra = Field(sequential = True, tokenize = tokenize, use_vocab=True, init_token="<sos>", eos_token="<eos>")
ita = Field(sequential = True, tokenize = tokenize, use_vocab=True, init_token="<sos>", eos_token="<eos>")
lat = Field(sequential= True, tokenize = tokenize, use_vocab=True, init_token="<sos>", eos_token="<eos>")

fields = {'spa':("spanish", spa), 'por':("portugese", por), 'rom':("romanian",rom),'fra':("french", fra),  "ita":('italian', ita), 'lat':("latin", lat)}

data = pd.read_csv("./data/rom-fra-ita-spa-por-lat.csv")

#set(data.ITALIAN.apply(list).sum())

train_data, test_data = TabularDataset.splits(
        path = './data',
        train = 'train.csv',
        test= 'test.csv',
        #validation = "valid.csv",
        format = 'csv',
        fields =fields)


spa.build_vocab(train_data, max_size=8799, min_freq=2, specials = ['<pad>'])
por.build_vocab(train_data, max_size=8799, min_freq=2, specials = ['<pad>'])
rom.build_vocab(train_data, max_size=8799, min_freq=2, specials = ['<pad>'])
fra.build_vocab(train_data, max_size=8799, min_freq=2, specials = ['<pad>'])
ita.build_vocab(train_data, max_size=8799, min_freq=2, specials = ['<pad>'])
lat.build_vocab(train_data, max_size=8799, min_freq=2, specials = ['<pad>'])

print(train_data[0].__dict__.keys())
print(train_data[1].__dict__.values())

ita_unique_sounds = {'-', 'a', 'b', 'd', 'e', 'f', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'z', 'ŋ', 'ɔ', 'ɛ', 'ɡ', 'ɲ', 'ɾ', 'ʃ', 'ʎ', 'ʒ', 'ː'}

lat_unique_sounds = {'-', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'y', 'z', 'ŋ', 'ɔ', 'ɛ', 'ɡ', 'ɪ', 'ɹ', 'ʊ', 'ʰ', 'ʷ', 'ː'}

rom_unique_sounds = {'-', 'a', 'b', 'd', 'e', 'f', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'y', 'z', 'ŋ', 'ɔ', 'ə', 'ɡ', 'ɨ', 'ɾ', 'ʃ', 'ʒ', 'ʲ'}

spa_unique_sounds = {'-', 'a', 'b', 'd', 'e', 'f', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'w', 'x', 'ð', 'ŋ', 'ɛ', 'ɡ', 'ɣ', 'ɪ', 'ɲ', 'ɾ', 'ʃ', 'ʊ', 'ʎ', 'ʝ', 'β', 'θ'}

por_unique_sounds = {'-', 'a', 'b', 'd', 'e', 'f', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'u', 'v', 'w', 'z', 'ŋ', 'ɐ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɡ', 'ɣ', 'ɨ', 'ɪ', 'ɲ', 'ɹ', 'ɾ', 'ʁ', 'ʃ', 'ʊ', 'ʎ', 'ʒ', '̃'}

fra_unique_sounds = {'-', 'a', 'b', 'd', 'e', 'f', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 's', 't', 'u', 'v', 'w', 'y', 'z', 'ø', 'œ', 'ɑ', 'ɔ', 'ə', 'ɛ', 'ɡ', 'ɲ', 'ʁ', 'ʃ', 'ʒ', 'ː', '̃'}



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
        print("Embedding:", embedding)
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states_ita, (hidden, cell) = self.rnn(embedding)
        print("encoder_states_ita", encoder_states_ita)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))
        print("x: ", x.shape)
        print("encoder states ita", encoder_states_ita.shape)
        return encoder_states_ita, hidden, cell    
    
  
class Encoder_Spa(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder_Spa, self).__init__()
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
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states_spa, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states_spa, hidden, cell

 
class Encoder_Fra(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder_Fra, self).__init__()
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
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states_fra, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states_fra, hidden, cell    
    
   
class Encoder_Rom(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder_Rom, self).__init__()
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
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states_rom, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states_rom, hidden, cell   
    
class Encoder_Por(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder_Por, self).__init__()
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
        # embedding shape: (seq_length, N, embedding_size)

        encoder_states_por, (hidden, cell) = self.rnn(embedding)
        # outputs shape: (seq_length, N, hidden_size)

        # Use forward, backward cells and hidden through a linear layer
        # so that it can be input to the decoder which is not bidirectional
        # Also using index slicing ([idx:idx+1]) to keep the dimension
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_cell(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states_por, hidden, cell   
    
    
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

    def forward(self, x, encoder_states_ita, encoder_states_spa, encoder_states_fra, 
                encoder_states_rom, encoder_states_por, hidden, cell):
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
        print(hidden) # convert hidden layers, unpack hidden layer tensors to text
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder1, encoder2, encoder3, encoder4, encoder5, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3
        self.encoder4 = encoder4
        self.encoder5 = encoder5
        self.decoder = decoder

    def forward(self, source1, source2, source3, source4, source5, target, teacher_force_ratio=0.5):
        batch_size = source1.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(lat.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)
        encoder_states_ita, hidden, cell = self.encoder1(source1)
        encoder_states_spa, hidden, cell = self.encoder2(source2)
        encoder_states_fra, hidden, cell = self.encoder3(source3)
        encoder_states_rom, hidden, cell = self.encoder4(source4)
        encoder_states_por, hidden, cell = self.encoder5(source5)

        # First input will be start of word token
        x = target[0]

        for t in range(1, target_len):
            # At every time step use encoder_states and update hidden, cell
            output, hidden, cell = self.decoder(x, encoder_states_ita, encoder_states_spa, encoder_states_fra, encoder_states_rom, encoder_states_por , hidden, cell)

            # Store prediction for current time step
            outputs[t] = output

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
num_epochs = 20
#The learning rate is a hyperparameter that controls how much to change the model in response to 
#the estimated error each time the model weights are updated
learning_rate = 0.001
batch_size = 64
        
#model hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size_encoder1 = len(ita_unique_sounds)
input_size_encoder2 = len(spa_unique_sounds)
input_size_encoder3 = len(fra_unique_sounds)
input_size_encoder4 = len(rom_unique_sounds)
input_size_encoder5 = len(por_unique_sounds)
input_size_decoder = len(lat_unique_sounds)
output_size = len(lat.vocab)
encoder_embedding_size_ita = 100
encoder_embedding_size_spa = 100
encoder_embedding_size_fra = 100
encoder_embedding_size_rom = 100
encoder_embedding_size_por = 100
decoder_embedding_size = 100
hidden_size = 1024
num_layers = 1 #if this is set to more layers I get a hidden[0] size error, apparently I set up sth weirdly
encoder_dropout = 0.5
decoder_dropout = 0.5


train_iterator, test_iterator = BucketIterator.splits((train_data, test_data), 
                                                      batch_size=batch_size,
                                                      #sort_within_batch = True,
                                                      #sort_key = lambda x: len(x.french),
                                                      device = device)
encoder_net1 = Encoder_Ita(input_size_encoder1, 
                      encoder_embedding_size_ita, 
                      hidden_size, 
                      num_layers, 
                      encoder_dropout).to(device)
encoder_net2 = Encoder_Spa(input_size_encoder2, 
                      encoder_embedding_size_spa, 
                      hidden_size, 
                      num_layers, 
                      encoder_dropout).to(device)
encoder_net3 = Encoder_Fra(input_size_encoder3, 
                      encoder_embedding_size_fra, 
                      hidden_size, 
                      num_layers, 
                      encoder_dropout).to(device)
encoder_net4 = Encoder_Rom(input_size_encoder4, 
                      encoder_embedding_size_rom, 
                      hidden_size, 
                      num_layers, 
                      encoder_dropout).to(device)
encoder_net5 = Encoder_Por(input_size_encoder5, 
                      encoder_embedding_size_por, 
                      hidden_size, 
                      num_layers, 
                      encoder_dropout).to(device)
decoder_net = Attention_Decoder(input_size_decoder, 
                      decoder_embedding_size, 
                      hidden_size,
                      output_size,
                      num_layers, 
                      decoder_dropout).to(device)                     
                      


model = Seq2Seq(encoder_net1, encoder_net2, encoder_net3, encoder_net4, encoder_net5, decoder_net)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#Tensorboard
writer = SummaryWriter(f"runs/loss_plot")
step = 0

pad_idx1 = ita.vocab.stoi["<pad>"]
pad_idx2 = spa.vocab.stoi["<pad>"]
pad_idx3 = fra.vocab.stoi["<pad>"]
pad_idx4 = rom.vocab.stoi["<pad>"]
pad_idx5 = por.vocab.stoi["<pad>"]
pad_idx6 = lat.vocab.stoi["<pad>"]
criterion = nn.NLLLoss(ignore_index=pad_idx1)

#minut	minyt	minuto	minuto	minutʊ	mɪnuːtʊm
#natsje	nasjɔ̃	natsione	naθjon	nɐsɐ̃ʊ̃	naːtɪoːnɛm
word1 = ("natsione")
word2 = ("naθjon")
word3 = ("nasjɔ̃")
word4 = ("nɐsɐ̃ʊ̃")
word5 = ("natsje")


for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")
    
    reconstructed_word = reconstruct_word(model, word1, word2, word3, word4, word5, device)

    with open('out.txt', 'a') as f:
        print(f"Reconstruct example words {word1, word2, word3, word4, word5,} \n as {reconstructed_word}", file=f)

    model.train()
    optimizer.zero_grad()
    model.eval()
    
    for batch_idx, element in enumerate(train_iterator):
                #Get input and targets
                #batch_idx is the iterator and elememt are the elemelon in train_iterator
                inp_data_spa = element.spanish
                inp_data_fra = element.french
                inp_data_ita = element.italian
                inp_data_por = element.portugese
                inp_data_rom = element.romanian
                targets = element.latin
    
                # Forward prop
                output = model(inp_data_fra, inp_data_ita, inp_data_por,
                                inp_data_rom, inp_data_spa, targets)
                print(output)
    
                # Output is of shape (trg_len, batch_size, output_dim) but NLLLoss
                # doesn't take input in that form  so we need to do some reshapin
                output = output[1:].reshape(-1, output.shape[2])
                targets = targets[1:].reshape(-1)
    
                optimizer.zero_grad()
                loss = criterion(output, targets)
    
                # Back prop
                loss.backward()
    
                # Clip to avoid exploding gradient issues, makes sure grads are in reasonable range
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    
                # Gradient descent step
                optimizer.step()
             
                #Calculate training accuracy
                _, predictions = output.max(1)
                num_correct = (predictions == targets).sum()
                running_training_accuracy = float(num_correct)//float(inp_data[0])
             
             
                # Plot to tensorboard
                writer.add_scalar("Training loss", loss, global_step=step)
                writer.add_scalar("Training Accuracy", running_training_accuracy, global_step=step)
                step += 1

