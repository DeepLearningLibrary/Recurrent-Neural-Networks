# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 15:42:27 2021

@author: Grant
"""

from res.sequential import EchoData
import torch
import torch.nn as nn
import torch.optim as optim

batch_size = 5
echo_step = 3
series_length = 20_000
BPTT_T = 20

train_data = EchoData(
    echo_step=echo_step,
    batch_size=batch_size,
    series_length=series_length,
    truncated_length=BPTT_T,
)
train_size = len(train_data)

test_data = EchoData(
    echo_step=echo_step,
    batch_size=batch_size,
    series_length=series_length,
    truncated_length=BPTT_T,
)
test_size = len(test_data)

class SimpleRNN(nn.Module):
    def __init__(self, input_size, rnn_hidden_size, output_size):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn = torch.nn.RNN(
            input_size=input_size,
            hidden_size=rnn_hidden_size,
            num_layers=1,
            nonlinearity='relu',
            batch_first=True
        )
        self.linear = torch.nn.Linear(
            in_features=rnn_hidden_size,
            out_features=1
        )

    def forward(self, x, hidden):
        x, hidden = self.rnn(x, hidden)  
        x = self.linear(x)
        return x, hidden

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(hidden):
    model.train()
       
    correct = 0
    for batch_idx in range(train_size):
        data, target = train_data[batch_idx]
        data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).float().to(device)
        optimizer.zero_grad()
        if hidden is not None: hidden.detach_()
        logits, hidden = model(data, hidden)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        
        pred = (torch.sigmoid(logits) > 0.5)
        correct += (pred == target.byte()).int().sum().item()
        
    return correct, loss.item(), hidden

def test(hidden):
    model.eval()   
    correct = 0
    with torch.no_grad():
        for batch_idx in range(test_size):
            data, target = test_data[batch_idx]
            data, target = torch.from_numpy(data).float().to(device), torch.from_numpy(target).float().to(device)
            logits, hidden = model(data, hidden)
            
            pred = (torch.sigmoid(logits) > 0.5)
            correct += (pred == target.byte()).int().sum().item()

    return correct

feature_dim = 1
h_units = 8

model = SimpleRNN(
    input_size=1,
    rnn_hidden_size=h_units,
    output_size=feature_dim
).to(device)
hidden = None
        
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

n_epochs = 5
epoch = 0

while epoch < n_epochs:
    correct, loss, hidden = train(hidden)
    epoch += 1
    train_accuracy = float(correct) / train_size
    print(f'Train Epoch: {epoch}/{n_epochs}, loss: {loss:.3f}, accuracy {train_accuracy:.1f}%')
   
correct = test(hidden)
test_accuracy = float(correct) / test_size
print(f'Test accuracy: {test_accuracy:.1f}%')