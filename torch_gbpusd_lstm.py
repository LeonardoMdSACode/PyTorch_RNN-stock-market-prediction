import numpy as np
import math

import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
import torch.optim as optim

import yfinance as yf
import mplfinance as mpf



df = yf.download("GBPUSD=X", start="1970-01-01")
# df.to_csv('GBPUSD.csv')
print(df)

print(df.shape)

print(df.describe())

# Define the plot style and type
mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
s = mpf.make_mpf_style(marketcolors=mc)
kwargs = dict(type='candle')

# Plot the data as a candlestick chart
mpf.plot(df, **kwargs, style=s, title='GBP/USD Exchange Rate')

df = df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'], axis=1)

print(df)

print(df.shape)

print(df.describe())

print(df.isnull().sum())

scaler = MinMaxScaler(feature_range=(-1, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1,1))

# function to create train, test data given stock data and sequence length
def load_data(stock, look_back):
    data_raw = stock.values # convert to numpy array
    data = []
    
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    
    data = np.array(data);
    test_set_size = int(np.round(0.2*data.shape[0]));
    train_set_size = data.shape[0] - (test_set_size);
    
    x_train = data[:train_set_size,:-1,:]
    y_train = data[:train_set_size,-1,:]
    
    x_test = data[train_set_size:,:-1]
    y_test = data[train_set_size:,-1,:]
    
    return [x_train, y_train, x_test, y_test]

look_back = 70 # choose sequence length
x_train, y_train, x_test, y_test = load_data(df, look_back)
print('x_train.shape = ',x_train.shape)
print('y_train.shape = ',y_train.shape)
print('x_test.shape = ',x_test.shape)
print('y_test.shape = ',y_test.shape)

# make training and test sets in torch
x_train = torch.from_numpy(x_train).type(torch.Tensor)
x_test = torch.from_numpy(x_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor)
y_test = torch.from_numpy(y_test).type(torch.Tensor)

y_train.size(),x_train.size()

# Showing the graph where training and test data divide
train_size = len(x_train)
test_size = len(x_test)

plt.figure(figsize=(10,5))
plt.title('GBP/USD Exchange Rate')
plt.plot(range(train_size), y_train, label='Training Data')
plt.plot(range(train_size, train_size + test_size), y_test, label='Test Data')
plt.xlabel('Time')
plt.ylabel('Scaled Closing Price')
plt.legend()

plt.axvline(x_train.shape[0], color='k', linestyle='--')

plt.show()

# Building model
input_dim = 1
hidden_dim = 16
num_layers = 2 
output_dim = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the model as a class
class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # (batch_dim, seq_dim, feature_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        # Readout layer
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()

        # To avoid backpropagation echo
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))

        # Index hidden state of last time step
        out = self.fc(out[:, -1, :]) 

        return out
    
model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss()

optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
print(model)
print(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    print(list(model.parameters())[i].size())
    
    
# Train model
num_epochs = 100
hist = {'MSE': [], 'val_loss': [], 'val_mae': [], 'val_mse': []}

# Number of steps to unroll
seq_dim =look_back-1  

for t in range(num_epochs):
    # Initialise hidden state to make LSTM stateless
    #model.hidden = model.init_hidden()
    
    # Forward pass
    y_train_pred = model(x_train)
    loss = loss_fn(y_train_pred, y_train)
    hist['MSE'].append(loss.item())
        
    y_val_pred = model(x_test)
    val_mae = torch.mean(torch.abs(y_test - y_val_pred))
    hist['val_mae'].append(val_mae.item())
    
    val_mse = torch.mean((y_test - y_val_pred) ** 2)
    hist['val_mse'].append(val_mse.item())


    if t % 10 == 0 and t !=0:
        print("Epoch ", t, "MSE: ", loss.item(), "Val_MAE: ", val_mae.item(), "Val_RMSE: ", val_mse.item())

    # Zero out gradient
    optimiser.zero_grad()

    # Backward pass
    loss.backward()

    # Update parameters
    optimiser.step()    
       
# Plot training loss
plt.plot(hist['MSE'], label="Training loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training loss')
plt.legend()
plt.show()

print(np.shape(y_train_pred))

# Predictions
y_test_pred = model(x_test)

# invert predictions
y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
y_train = scaler.inverse_transform(y_train.detach().numpy())
y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
y_test = scaler.inverse_transform(y_test.detach().numpy())

# calculate RMSE
trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
print('Train Score: %.5f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
print('Test Score: %.5f RMSE' % (testScore))
#LSTM Train Score: 0.01443 RMSE
# Test Score: 0.01414 RMSE

# Plotting the results
figure, axes = plt.subplots(figsize=(15, 6))
axes.xaxis_date()

axes.plot(df[len(df)-len(y_test):].index, y_test, color = 'red', label = 'Real GBP/USD Stock Price')
axes.plot(df[len(df)-len(y_test):].index, y_test_pred, color = 'blue', label = 'Predicted GBP/USD Stock Price')

plt.title('GBP/USD Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('GBP/USD Stock Price')
plt.legend()
plt.show()
