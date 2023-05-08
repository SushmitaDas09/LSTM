import numpy as np
import torch
from lstm_model import LSTMModel
from dataloader import SalesDataset
import pandas as pd

if __name__ == "__main__":
    timesteps = 30
    train_sales_stats = pd.read_csv("train_sales_stats.csv")
    mean, std = train_sales_stats["mean"][0], train_sales_stats["std"][0]
    
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    num_stores = len(train_df['store_nbr'].unique())
    num_families = len(train_df['family'].unique())
    rows_per_timestep = num_stores * num_families

    train_arr = train_df[["sales", "onpromotion"]].to_numpy().reshape(-1, num_stores, num_families, 2)
    test_arr = test_df["onpromotion"].to_numpy().reshape(-1, num_stores, num_families)
    num_preds = test_arr.shape[0]

    # timesteps, n_stores, n_families -> 30, n_stores * n_families
    sales = (train_arr[-timesteps:, :, :, 0] - mean) / std
    sales = sales.reshape((timesteps, num_stores, num_families))
    onpromotion = np.concatenate((train_arr[-timesteps+1:, :, :, 1], test_arr))
    onpromotion = onpromotion.reshape((timesteps - 1 + num_preds, num_stores, num_families))

    model = LSTMModel(input_size=(num_stores * num_families * 2), hidden_size = 1024, output_size=num_stores * num_families)
    model.load_state_dict(torch.load('9.pt'))
    h0 = torch.zeros(1, 1, 1024)
    c0 = torch.zeros(1, 1, 1024)
    predictions = []
    for i in range(num_preds):
        sales_i = np.expand_dims(sales, axis=3)
        onpromotion_i = np.expand_dims(onpromotion[i: i + timesteps], axis=3)
        inp = torch.tensor(np.concatenate((sales_i, onpromotion_i), axis = 3).reshape(1, timesteps, rows_per_timestep * 2), dtype = torch.float)
        out, hidden = model.inference(inp, h0, c0)
        out = out.detach().cpu().numpy()

        predictions.append(out)
        sales = np.concatenate((sales[:-1], out.reshape((1, num_stores, num_families))), axis = 0)
    
    predictions = np.array(predictions).flatten()
    predictions = mean + (predictions * std)
    test_df["sales"] = np.clip(predictions, a_min=0, a_max=None)
    test_df = test_df[["id", "sales"]]
    test_df.to_csv("lstm_30_10_epochs.csv", index=False)

  


