import numpy as np
import torch
from lstm_model import LSTMModel
from dataloader import SalesDataset
import pandas as pd

if __name__ == "__main__":
    timesteps = 60
    # train_sales_stats = pd.read_csv("train_sales_stats.csv")
    train_df = pd.read_csv("train_sort.csv")
    test_df = pd.read_csv("test_sort.csv")

    num_stores = len(train_df['store_nbr'].unique())
    num_families = len(train_df['family'].unique())
    rows_per_timestep = num_stores * num_families

    train_arr = train_df[["normalized_sales", "onpromotion"]].to_numpy().reshape(-1, num_stores, num_families, 2)
    test_arr = test_df["onpromotion"].to_numpy().reshape(-1, num_stores, num_families)
    num_preds = test_arr.shape[0]

    # timesteps, n_stores, n_families -> 30, n_stores * n_families
    sales = train_arr[-timesteps:, :, :, 0]
    sales = sales.reshape((timesteps, num_stores, num_families))
    onpromotion = np.concatenate((train_arr[-timesteps+1:, :, :, 1], test_arr))
    onpromotion = onpromotion.reshape((timesteps - 1 + num_preds, num_stores, num_families))

    model = LSTMModel(input_size=(num_stores * num_families * 2), hidden_size = 2048, output_size=num_stores * num_families)
    model.load_state_dict(torch.load('lstm_sort/9.pt'))
    h = torch.zeros(1, 1, 2048)
    c = torch.zeros(1, 1, 2048)
    predictions = []
    for i in range(num_preds):
        sales_i = np.expand_dims(sales, axis=3)
        onpromotion_i = np.expand_dims(onpromotion[i: i + timesteps], axis=3)
        inp = torch.tensor(np.concatenate((sales_i, onpromotion_i), axis = 3).reshape(1, timesteps, rows_per_timestep * 2), dtype = torch.float)
        out, hidden = model.inference(inp, h, c)
        out = out.detach().cpu().numpy()
  
        predictions.append(out)
        sales = np.concatenate((sales[1:], out.reshape((1, num_stores, num_families))), axis = 0)
        h,c = hidden
    predictions = np.array(predictions).flatten()
    result_df = pd.DataFrame(test_df[["id", "store_nbr", "family"]])
    result_df["normalized_sales"] = predictions
    result_df.to_csv("lstm_sort_60_9_epochs.csv", index=False)

  


