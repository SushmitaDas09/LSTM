import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from lstm_model import LSTMModel
from dataloader import SalesDataset
import pandas as pd


if __name__ == "__main__":
    ts_data = SalesDataset("train.csv", timesteps=30, normalize=False)
    # pd.DataFrame({"mean": [ts_data.mean], "std": [ts_data.std]}).to_csv("train_sales_stats.csv")
    
    x_shape, y_shape = ts_data[0][0].shape, ts_data[0][1].shape
    _, in_feats = x_shape
    model = LSTMModel(input_size=in_feats, hidden_size=1024, output_size=y_shape[0])

    train_size = int(0.8 * len(ts_data))
    val_size = len(ts_data) - train_size

    # Split your dataset into train and validation sets
    train_dataset, val_dataset = random_split(ts_data, [train_size, val_size])

    # Create your PyTorch dataloaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


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
            loss = torch.sqrt(criterion(output, target))
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