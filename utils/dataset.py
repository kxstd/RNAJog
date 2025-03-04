from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from utils.random_gen import cod2case
import pandas as pd


class proDataset_padding(Dataset):
    def __init__(self, data_path_list, label_get=False):
        dataset = pd.concat([pd.read_csv(data_path) for data_path in data_path_list])
        max_length = 3066
        self.data = dataset["Sequence"][dataset["Length"] <= max_length]
        self.label_get = label_get
        if label_get:
            self.label = dataset["Value"] 

        self.data = self.data.str.pad(max_length, side="right", fillchar="?")
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.label_get:
            return self.data.iloc[idx], self.label.iloc[idx]
        
        return self.data.iloc[idx]



def load_data(data_path_list, sampler_type, batch_size, num_batch, num_workers=1):
    dataset = proDataset_padding(data_path_list)
    if sampler_type == "Sequencial":
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    elif sampler_type == "Random":
        rand_sampler = RandomSampler(dataset, replacement=True, num_samples = batch_size*num_batch)
        batch_sampler = BatchSampler(rand_sampler, batch_size, drop_last=True)
        dataloader = DataLoader(dataset, num_workers=num_workers, sampler = batch_sampler, collate_fn=lambda x: x[0])    
    else:
        raise ValueError("Invalid sampler type: {}".format(sampler_type))
    return dataloader
    

    
