import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SalesDataset(Dataset):
    def __init__(self, csv_file, timesteps, use_last_k = -1):
        self.sales_df = pd.read_csv(csv_file)
        self.timesteps = timesteps
        self.num_stores = len(self.sales_df['store_nbr'].unique())
        self.num_families = len(self.sales_df['family'].unique())
        self.rows_per_timestep = self.num_stores * self.num_families
        
        if use_last_k > 0:
            self.sales_df = self.sales_df.iloc[-use_last_k * self.rows_per_timestep:]


        # use current onpromotion value
        self.sales_df['onpromotion'] = self.sales_df['onpromotion'].shift(-1)
        feat_keys = ['sales', 'onpromotion']
        # weekdays = ['Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        # weekdays_list = [f"weekday_{day}" for day in weekdays]
        # feat_keys += weekdays_list
        self.num_feats = len(feat_keys)
        self.sales_df = self.sales_df[feat_keys]
        
    def __len__(self):
        assert len(self.sales_df) % self.rows_per_timestep == 0
        return len(self.sales_df) // self.rows_per_timestep - self.timesteps
    
    def __getitem__(self, idx):
        # Select the sales and onpromotion data for the last `timesteps` time steps
        start_idx = idx * self.rows_per_timestep
        end_idx = (idx + self.timesteps + 1) * self.rows_per_timestep
        window_df = self.sales_df.iloc[start_idx:end_idx, :]
        features = window_df.to_numpy().reshape((self.timesteps + 1, self.rows_per_timestep, self.num_feats))
        x = features[:-1].reshape((self.timesteps, self.rows_per_timestep * self.num_feats))
        y = features[-1, :, 0].reshape(self.rows_per_timestep)
        
        # Convert to PyTorch tensor and return
        return torch.from_numpy(x).float(), torch.from_numpy(y).float()

# Example usage:
# dataset = SalesDataset('train.csv', timesteps=7)
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
# print('ehllo')
# for inputs, targets in dataloader:
#     print('Input shape:', inputs.shape)
#     print('Target shape:', targets.shape)
#     break
