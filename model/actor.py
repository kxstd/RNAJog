import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
# from utils.constant import MASK, trans_dict, pro2id, id2pro
# from utils.evaluate import calculate_cai_single, get_cai_prob
import torch.nn.functional as F
import math
from utils.tokenizer import get_tokenizer
from utils.constant import codonlist, get_mask, pro2id, trans_idx2codon, GC_CONTENT

# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0,d_model,2)*-(math.log(10000.0)/d_model))
#         pe[:,0::2] = torch.sin(position*div_term)
#         pe[:,1::2] = torch.cos(position*div_term)
#         pe = pe.unsqueeze(0).transpose(0,1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         x = x + self.pe[:x.size(0), :]
#         return self.dropout(x)
    




class Actor(nn.Module):
    def __init__(self,cfg,device):
        super(Actor,self).__init__()


        # self.cnn = nn.Conv1d(cfg.embedder.embedding_size, cfg.actor.cnn.hidden_size, 1)
        self.codonlist = ["&&&"] + codonlist
        self.codon2id = dict((c,i) for i,c in enumerate(self.codonlist))
        # self.embedder = nn.Embedding(65, cfg.embedder.embedding_size)
        # self.embedder = Embedder(cfg)
        self.hidden_size = cfg.actor.hidden_size
        self.context_size = cfg.actor.context_size
        self.l1 = nn.Linear(cfg.embedder.embedding_size, self.hidden_size*6)
        # self.positional_encoding = PositionalEncoding(cfg.embedder.embedding_size)
        # self.decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.embedder.embedding_size, nhead=cfg.actor.transformer.nhead)
        # self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=cfg.actor.transformer.num_layers)
        
        self.predictor = nn.Sequential(nn.Linear(self.hidden_size + self.context_size, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, 1),
                                       nn.ReLU())
        self.softmax = nn.Softmax(dim=-1)
        self.lstm = nn.LSTMCell(self.hidden_size + self.context_size, self.context_size)
        self.device = torch.device(device)
        # self.rnn = nn.LSTM(
        #     input_size=cfg.embedder.embedding_size,
        #     hidden_size=cfg.actor.lstm.hidden_size,
        #     num_layers=cfg.actor.lstm.num_layers,
        #     batch_first=True
        # )
        # self.out = nn.Conv1d(cfg.actor.lstm.hidden_size,6,1)
        # self.base_prob_weight = cfg.base_prob_weight

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
        choice_features = F.relu(self.l1(memory[:,1:-1,:]).view(memory.shape[0], memory.shape[1]-2, 6, self.hidden_size)) # [batch_size, seq_len, 6, hidden_size]
        h = torch.zeros(memory.shape[0], self.context_size).to(self.device)
        c = torch.zeros(memory.shape[0], self.context_size).to(self.device)
        
        for t in range(1,max_length):
            # print("{}/{}".format(t,max_length))
            
            choice_features_t = torch.cat([choice_features[:,t,:,:], h.unsqueeze(1).repeat(1,6,1)], dim=-1) # [batch_size, 6, hidden_size + context_size]
            scores = self.predictor(choice_features_t).squeeze(-1) # [batch_size, 6]

            # scores = torch.where(torch.isnan(scores), torch.zeros_like(scores), scores)

            prob = self.softmax(torch.where(mask[:,t,:], scores, torch.tensor(float('-inf')).to(mask.device)))
            
            if merge_prob is not None:
                prob = prob * alpha + merge_prob[:,t,:] * (1-alpha)
            

            distr = Categorical(probs = prob) # [batch_size]
            
            
            idx = distr.sample() # [batch_size]
            logs = distr.log_prob(idx) # [batch_size]
            chosen_features_t = torch.stack([choice_features_t[i, idx[i], :] for i in range(memory.shape[0])]) # [batch_size, hidden_size + context_size]
            
            h, c = self.lstm(chosen_features_t, (h, c))

            for i in range(len(log_probs_list)):
                log_probs_list[i].append(logs[i] if length[i] > t else torch.tensor(0.0).to(self.device))
    
            output_t = trans_idx2codon([[x] for x in idx], [pro[i][t] for i in range(len(pro))])
            codons = [codons[i] + output_t[i] for i in range(len(codons))]
            
        log_probs = torch.stack([sum(log_probs_list[i]) for i in range(len(log_probs_list))])
        return codons, log_probs
    

    def optimize(self, memory, pro, length, merge_prob = None, alpha = None, sample_method = "greedy", sample_temperature = 1, ban_table=None, gc_repress = 1):
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
        choice_features = F.relu(self.l1(memory[:,1:-1,:]).view(memory.shape[0], memory.shape[1]-2, 6, self.hidden_size)) # [batch_size, seq_len, 6, hidden_size]
        h = torch.zeros(memory.shape[0], self.context_size).to(self.device)
        c = torch.zeros(memory.shape[0], self.context_size).to(self.device)
        for t in range(1,max_length):
            choice_features_t = torch.cat([choice_features[:,t,:,:], h.unsqueeze(1).repeat(1,6,1)], dim=-1) # [batch_size, 6, hidden_size + context_size]
            scores = self.predictor(choice_features_t).squeeze(-1) # [batch_size, 6]
            prob = self.softmax(torch.where(mask[:,t,:], scores, torch.tensor(float('-inf')).to(mask.device)))

            if merge_prob is not None:
                prob = prob * alpha + merge_prob[:,t,:] * (1-alpha)

            if gc_repress is not None:
                for i in range(memory.shape[0]):
                    gc_content = GC_CONTENT[pro[i][t]].to(self.device) # get the gc content of the current protein

                    gc_repress_vector = gc_content ** gc_repress
                    prob[i] = prob[i] * gc_repress_vector
                    if prob[i].sum() == 0:
                        raise ValueError("All codons are banned by GC repression")
                    prob[i] = prob[i] / prob[i].sum()

            prob = self.softmax(torch.where(mask[:,t,:], prob, torch.tensor(float('-inf')).to(mask.device))/sample_temperature)

            # prob modify for ban subsequences
            if ban_table is not None:
                for i in range(memory.shape[0]):
                    # if codons[i].endswith("CUC") and pro[i][t] == "E":
                    #     print(codons[i][-len("CUC"):])
                    #     print("".join(pro[i][t:]))
                    for key, value in ban_table.items():
                        prefix, pro_tail = key
                        # if the tail of codons[i] is prefix and the head of pro[i][t:] is pro_tail
                        if codons[i][len(codons[i])-len(prefix):] == prefix and "".join(pro[i][t:]).startswith(pro_tail):       
                            prob[i][value] = 0
                            # print(print(value))
                            # print(prob[i])
                            if prob[i].sum() == 0:
                                raise ValueError("All codons are banned")
                            prob[i] = prob[i] / prob[i].sum()

            if sample_method == "greedy":
                idx = torch.argmax(prob, dim=-1)
            elif sample_method == "sample":
                distr = Categorical(probs = prob) # [batch_size]
                idx = distr.sample() # [batch_size]
            else:
                raise ValueError("Invalid sample method: {}".format(sample_method))
            
            chosen_features_t = torch.stack([choice_features_t[i, idx[i], :] for i in range(memory.shape[0])]) # [batch_size, hidden_size + context_size]
            
            h, c = self.lstm(chosen_features_t, (h, c))
    
            output_t = trans_idx2codon([[x] for x in idx], [pro[i][t] for i in range(len(pro))])
            codons = [codons[i] + output_t[i] for i in range(len(codons))]
        
        return codons, None
    

        
                


