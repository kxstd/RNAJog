from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, BatchSampler
from utils.random_gen import cod2case
import pandas as pd


class proDataset_padding(Dataset):
    def __init__(self, data_path_list, label_get=False):
        dataset = pd.concat([pd.read_csv(data_path) for data_path in data_path_list])
        max_length = 3066
        # max_length = 1800
        self.data = dataset["Sequence"][dataset["Length"] <= max_length]
        self.label_get = label_get
        if label_get:
            self.label = dataset["Value"] 

        # padding the sequence to the same length
        

        # 加速优化下面的padding代码，用pandas的str.pad方法
        # for i in range(len(self.data)):
        #     if len(self.data[i]) < max_length:
        #         self.data[i] = self.data[i] + "[P]" * (max_length - len(self.data[i]))
            
        self.data = self.data.str.pad(max_length, side="right", fillchar="?")
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.label_get:
            return self.data.iloc[idx], self.label.iloc[idx]
        
        return self.data.iloc[idx]



# class proDataset(Dataset):
#     def __init__(self, data_path_list, split):
#         dataset = pd.concat([pd.read_csv(data_path) for data_path in data_path_list])
#         self.data = dataset["Sequence"]
#         # codon_seqs, amino_seqs = pro2case(0, dataset["Sequence"][:100])
#         # codon_seqs, amino_seqs = cod2case(0, dataset)
#         # self.data = pd.DataFrame({"codon":codon_seqs, "amino":amino_seqs})

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data.iloc[idx]

# class proDataset_fix_length(Dataset):
#     def __init__(self, data_path_list, split, length):
#         dataset = pd.concat([pd.read_csv(data_path) for data_path in data_path_list])
#         self.data = dataset[dataset["Length"] == length]["Sequence"]
        

#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         return self.data.iloc[idx]


def load_data(data_path_list, sampler_type, batch_size, num_batch, num_workers=1):
    # if length == -1:
    #     dataset = proDataset(data_path_list, split)
    #     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    #     return dataloader
    
    # dataset = proDataset_fix_length(data_path_list, split, length)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
    

    
