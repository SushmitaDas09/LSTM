import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

class TimeSeriesDataset(Dataset):
    def __init__(self, data_path, timesteps):
        data = pd.read_csv(data_path)
        data_list = []
        grouped = data.groupby(['family', 'store_nbr'])

        for name, group in grouped:
            data = group[['date', 'sales', 'store_nbr', 'onpromotion']].dropna()
            data_list.append(data.set_index('date'))
        self.data = data_list
        means, stds = [], [] 
        for i, df in enumerate(self.data):
            means.append(df['sales'].mean())
            stds.append(df['sales'].std())
        self.mean = np.mean(means)
        self.std = np.mean(stds)
        for i, df in enumerate(self.data):
            df['sales'] = (df['sales'] - self.mean)/self.std

        self.timesteps = timesteps

    def __len__(self):
        return sum(len(df) - self.timesteps - 1 for df in self.data)

    def __getitem__(self, idx):
        for i, df in enumerate(self.data):
            if idx < len(df) - self.timesteps - 1:
                x = torch.tensor(df.iloc[idx:idx+self.timesteps][['sales', 'onpromotion']].values)
                y = torch.tensor(df.iloc[idx+self.timesteps]['sales'])
                return x.flatten().float(), y.reshape(1,).float()
            else:
                idx -= len(df) - self.timesteps - 1

# Example usage
timesteps = 30  # the number of timesteps to use for each sample

ts_data = TimeSeriesDataset("train.csv", timesteps)
df = pd.DataFrame({"mean": [ts_data.mean], "std": [ts_data.std]})
df.to_csv("sales_stats.csv", index=False)

model = nn.Sequential(nn.Linear(timesteps * 2, 256),
                    nn.LeakyReLU(),
                    nn.Linear(256, 32),
                    nn.LeakyReLU(),
                    nn.Linear(32, 1),
                    nn.LeakyReLU()).float()


train_size = int(0.8 * len(ts_data))
val_size = len(ts_data) - train_size

# Split your dataset into train and validation sets
train_dataset, val_dataset = random_split(ts_data, [train_size, val_size])

# Create your PyTorch dataloaders for train and validation sets
train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)


# Define your model and loss function
criterion = nn.MSELoss()

# Define your optimizer
optimizer = optim.Adam(model.parameters())

# Train loop
n_epochs = 10
for epoch in range(n_epochs):
    train_loss = 0.0
    val_loss = 0.0
    
    # Training
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        print(loss)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss /= len(train_loader.dataset)
    
    # Validation
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            output = model(data)
            loss = criterion(output, target)
            val_loss += loss.item() * data.size(0)
        val_loss /= len(val_loader.dataset)
    
    torch.save(model.state_dict(), f"{epoch}.pt")
    # Print train and validation loss for this epoch
    print(f"Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
