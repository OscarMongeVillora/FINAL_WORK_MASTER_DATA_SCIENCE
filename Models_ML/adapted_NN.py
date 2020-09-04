import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

class NN_model(object):
    def __init__(self, list_hyperparam_model, param_epoch, train_window):
        #self.train_window, self.lr = list_hyperparam_model
        self.lr = list_hyperparam_model[0]
        self.epochs = param_epoch
        self.train_window = train_window

    # DIVIDE THE DATA_TRAIN IN SUBSETS OF "XTRAIN (PREVIOUS VALUES)" AND "YTRAIN (PREDICTION)"
    def create_inout_sequences(self, input_data, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw):
            train_seq = input_data[i:i + tw]
            train_label = input_data[i + tw:i + tw + 1]
            inout_seq.append((train_seq, train_label))
        return inout_seq


    def fit(self, x_features, data_train):

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        data_train = np.array(data_train)
        self.train_data_normalized = self.scaler.fit_transform(data_train.reshape(-1, 1))
        #self.train_data_normalized = torch.from_numpy(train_data_normalized).float()
        self.train_data_normalized_tensor = torch.FloatTensor(self.train_data_normalized).view(-1)
        train_inout_seq = self.create_inout_sequences(self.train_data_normalized_tensor, int(self.train_window))

        self.model = LSTM()
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        epochs = self.epochs
        single_loss = None

        for i in range(epochs):
            for seq, labels in train_inout_seq:
                optimizer.zero_grad()
                self.model.hidden_cell = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                     torch.zeros(1, 1, self.model.hidden_layer_size))

                y_pred = self.model(seq)

                single_loss = loss_function(y_pred, labels)
                single_loss.backward()
                optimizer.step()

            if i % 25 == 1:
                print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

        print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


    def predict(self, x_features):

        fut_pred = len(x_features)
        test_inputs = self.train_data_normalized.squeeze()[-int(self.train_window):].tolist()

        for i in range(fut_pred):
            seq = torch.FloatTensor(np.array(test_inputs).reshape(-1,1))
            with torch.no_grad():
                self.model.hidden = (torch.zeros(1, 1, self.model.hidden_layer_size),
                                    torch.zeros(1, 1, self.model.hidden_layer_size))
                test_inputs.append(self.model(seq).item())

        actual_predictions = self.scaler.inverse_transform(np.array(test_inputs[int(self.train_window):]).reshape(-1, 1))

        return actual_predictions


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

