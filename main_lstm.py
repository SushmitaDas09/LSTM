import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from lstm_model import LSTMModel
from dataloader import SalesDataset
from test_lstm import test
import pandas as pd

def rmsle_loss(y_pred, y_true):
    return torch.sqrt(torch.mean((torch.log(1 + y_pred) - torch.log(1 + y_true)) ** 2))

def train(sales_df, max_epochs, log_dir, min_epochs = 3):
    ts_data = SalesDataset(sales_df, timesteps=60)
    # pd.DataFrame({"mean": [ts_data.mean], "std": [ts_data.std]}).to_csv("train_sales_stats.csv")
    
    x_shape, y_shape = ts_data[0][0].shape, ts_data[0][1].shape
    _, in_feats = x_shape
    model = LSTMModel(input_size=in_feats, hidden_size=512, output_size=y_shape[0])

    train_size = int(0.9 * len(ts_data))
    val_size = len(ts_data) - train_size

    # Split your dataset into train and validation sets
    train_dataset, val_dataset = random_split(ts_data, [train_size, val_size])

    # Create your PyTorch dataloaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


    # Define your model and loss function
    mse_criterion = nn.MSELoss()
    rmsle_criterion = rmsle_loss
    

    # Define your optimizer
    optimizer = optim.Adam(model.parameters())

    final_val_loss = 0.0
    # Train loop
    for epoch in range(max_epochs):
        train_loss = 0.0
        val_loss_mse = 0.0
        val_loss_rmsle = 0.0
        
        # Training
        model.train()
        for data, target in train_loader:
            optimizer.zero_grad()
            output = model(data)
            loss = rmsle_criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        with torch.no_grad():
            for data, target in val_loader:
                output = model(data)
                loss_mse = mse_criterion(output, target)
                loss_rmsle = rmsle_criterion(output, target)
                val_loss_mse += loss_mse.item() * data.size(0)
                val_loss_rmsle += loss_rmsle.item() * data.size(0)
            val_loss_mse /= len(val_loader.dataset)
            val_loss_rmsle /= len(val_loader.dataset)
            final_val_loss = val_loss_rmsle
        os.makedirs(log_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{log_dir}/{epoch}.pt")
        # Print train and validation loss for this epoch
        print(f"Epoch {epoch+1}/{max_epochs}, Train Loss: {train_loss:.4f}, Val Loss MSE: {val_loss_mse:.4f}, Val Loss RMSLE: {val_loss_rmsle:.4f}")
        if epoch + 1 >= min_epochs and val_loss_rmsle < 0.2:
            break
    return model, final_val_loss

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
families = train_data['family'].unique()
families.sort()
model_root_dir = "family_models"
result_df_list = []
avg_val_list = []
for fid, family in enumerate(families):
    print(f"{family} {fid+1}/{len(families)}")
    filtered_train_data = train_data[train_data['family']==family]
    filtered_train_data = filtered_train_data.sort_values(by = ['date','store_nbr'])
    filtered_test_data = test_data[test_data['family']==family]
    filtered_test_data = filtered_test_data.sort_values(by = ['date','store_nbr'])
    # target_scaler = MinMaxScaler()
    # target_scaler.fit(train_data['sales'].values.reshape(-1, 1))
    # y_train_norm= target_scaler.transform(train_data['sales'].values.reshape(-1, 1))

    # y_pred = target_scaler.inverse_transform(train_data['normalized_sales'].values.reshape(-1,1))
    
    model, val_loss = train(filtered_train_data, 10, os.path.join(model_root_dir, family))
    avg_val_list.append(val_loss)
    results = test(filtered_train_data, filtered_test_data, model)
    result_df_list.append(results)
result_df = pd.concat(result_df_list, axis=0)
result_df = result_df[["id", "sales"]]
result_df.to_csv("family_lstm_submission.csv", index=False)
print("Average val loss", np.mean(avg_val_list))