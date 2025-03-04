import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import torch.nn.functional as F
from utils.constant import codonlist, get_mask, trans_idx2codon


class Actor(nn.Module):
    def __init__(self,cfg):
        super(Actor,self).__init__()


        self.codonlist = ["&&&"] + codonlist
        self.codon2id = dict((c,i) for i,c in enumerate(self.codonlist))
        self.hidden_size = cfg.actor.hidden_size
        self.context_size = cfg.actor.context_size
        self.l1 = nn.Linear(cfg.embedder.embedding_size, self.hidden_size*6)
        self.predictor = nn.Sequential(nn.Linear(self.hidden_size + self.context_size, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, 1),
                                       nn.ReLU())
        self.softmax = nn.Softmax(dim=-1)
        self.lstm = nn.LSTMCell(self.hidden_size + self.context_size, self.context_size)
        self.device = torch.device(cfg.device)

    def forward(self, memory, pro, length, merge_prob = None, alpha = None):
        '''
        memory: [batch_size, seq_len+2, embedding_size]
        pro: [batch_size, seq_len]
        length: [batch_size]
        merge_prob: [batch_size, seq_len, 6]
        alpha: A number between 0 and 1
        '''    
        length = torch.tensor(length).to(self.device)
        max_length = length.max()
        mask = get_mask(pro).to(self.device)
        log_probs_list = [[torch.tensor(0.0).to(self.device)] for i in range(memory.shape[0])]

        codons = ["AUG" for i in range(memory.shape[0])]
        choice_features = F.relu(self.l1(memory[:,1:-1,:]).view(memory.shape[0], memory.shape[1]-2, 6, self.hidden_size)) 
        h = torch.zeros(memory.shape[0], self.context_size).to(self.device)
        c = torch.zeros(memory.shape[0], self.context_size).to(self.device)
        
        for t in range(1,max_length):            
            choice_features_t = torch.cat([choice_features[:,t,:,:], h.unsqueeze(1).repeat(1,6,1)], dim=-1) 
            scores = self.predictor(choice_features_t).squeeze(-1)
            prob = self.softmax(torch.where(mask[:,t,:], scores, torch.tensor(float('-inf')).to(mask.device)))
            
            if merge_prob is not None:
                prob = prob * alpha + merge_prob[:,t,:] * (1-alpha)
            
            distr = Categorical(probs = prob) 
                    
            idx = distr.sample() 
            logs = distr.log_prob(idx) 
            chosen_features_t = torch.stack([choice_features_t[i, idx[i], :] for i in range(memory.shape[0])])
            
            h, c = self.lstm(chosen_features_t, (h, c))

            for i in range(len(log_probs_list)):
                log_probs_list[i].append(logs[i] if length[i] > t else torch.tensor(0.0).to(self.device))
    
            output_t = trans_idx2codon([[x] for x in idx], [pro[i][t] for i in range(len(pro))])
            codons = [codons[i] + output_t[i] for i in range(len(codons))]
            
        log_probs = torch.stack([sum(log_probs_list[i]) for i in range(len(log_probs_list))])
        return codons, log_probs
    

    def optimize(self, memory, pro, length, merge_prob = None, alpha = None, sample_method = "greedy", sample_temperature = 1, ban_table=None):
        '''
        Only support same length protein sequence in one batch.

        memory: [batch_size, seq_len+2, embedding_size]
        pro: [batch_size, seq_len]
        length: [batch_size]
        merge_prob: [batch_size, seq_len, 6]
        alpha: A number between 0 and 1
        ban_table:{(pre_seq, next_pro):[ban_codon_mask]}
        '''    
        length = torch.tensor(length).to(self.device)
        max_length = length.max()
        mask = get_mask(pro).to(self.device)
        codons = ["AUG" for i in range(memory.shape[0])]
        choice_features = F.relu(self.l1(memory[:,1:-1,:]).view(memory.shape[0], memory.shape[1]-2, 6, self.hidden_size)) 
        h = torch.zeros(memory.shape[0], self.context_size).to(self.device)
        c = torch.zeros(memory.shape[0], self.context_size).to(self.device)
        
        for t in range(1,max_length):
            choice_features_t = torch.cat([choice_features[:,t,:,:], h.unsqueeze(1).repeat(1,6,1)], dim=-1) 
            scores = self.predictor(choice_features_t).squeeze(-1) 
            prob = self.softmax(torch.where(mask[:,t,:], scores, torch.tensor(float('-inf')).to(mask.device)))

            if merge_prob is not None:
                prob = prob * alpha + merge_prob[:,t,:] * (1-alpha)
                prob = self.softmax(torch.where(mask[:,t,:], prob, torch.tensor(float('-inf')).to(mask.device))/sample_temperature)
            
            if ban_table is not None:
                for i in range(memory.shape[0]):
                    for key, value in ban_table.items():
                        prefix, pro_tail = key
                        if codons[i][-len(prefix):] == prefix and "".join(pro[i][t:]).startswith(pro_tail):       
                            prob[i][value] = 0
                            if prob[i].sum() == 0:
                                raise ValueError("All codons are banned")
                            prob[i] = prob[i] / prob[i].sum()

            if sample_method == "greedy":
                idx = torch.argmax(prob, dim=-1)
            elif sample_method == "sample":
                distr = Categorical(probs = prob) 
                idx = distr.sample() 
            else:
                raise ValueError("Invalid sample method: {}".format(sample_method))
            
            chosen_features_t = torch.stack([choice_features_t[i, idx[i], :] for i in range(memory.shape[0])])
            
            h, c = self.lstm(chosen_features_t, (h, c))
    
            output_t = trans_idx2codon([[x] for x in idx], [pro[i][t] for i in range(len(pro))])
            codons = [codons[i] + output_t[i] for i in range(len(codons))]
        
        return codons, None
    

        
                


