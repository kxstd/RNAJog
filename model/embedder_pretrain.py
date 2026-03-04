import numpy as np
import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from utils.tokenizer import get_tokenizer

class Embedder(nn.Module):
    def __init__(self,cfg,device):
        super(Embedder,self).__init__()
        self.tokenizer = get_tokenizer()
        self.model = BertForSequenceClassification.from_pretrained(cfg.embedder.model_dir)
        self.model.eval()
        self.cuda = (device == "cuda")
    def forward(self,seqs):
        output_embeds = []
        for seq in seqs:
            input_ids = self.tokenizer.encode(" ".join(seq)).ids
            input_ids = torch.tensor([input_ids], dtype=torch.int64)
            if self.cuda:
                input_ids = input_ids.cuda()
            with torch.no_grad():
                input_ids_back = input_ids
                hidden_states_list = []
                while input_ids_back.shape[1] > 1024:
                    input_ids_front = input_ids_back[:,:1024]
                    input_ids_back = input_ids_back[:,1024:]
                    outputs_front = self.model(input_ids_front, labels=None, output_hidden_states=True)
                    _, hidden_states_front = outputs_front[:3]
                    hidden_states_list.append(hidden_states_front)
                outputs_back = self.model(input_ids_back, labels=None, output_hidden_states=True)
                _, hidden_states_back = outputs_back[:3]
                hidden_states_list.append(hidden_states_back)
                output_embed = torch.cat([torch.squeeze(hidden_states[-1]) for hidden_states in hidden_states_list],dim=0)
                output_embeds.append(output_embed)
        output_embeds = torch.stack(output_embeds)
        #         if input_ids.shape[1]>1024:
        #             input_ids_front = input_ids[:,:1024]
        #             input_ids_back = input_ids[:,1024:]
        #             outputs_front = self.model(input_ids_front, labels=None, output_hidden_states=True)
        #             _, hidden_states_front = outputs_front[:3]
        #             output_back = self.model(input_ids_back, labels=None, output_hidden_states=True)
        #             _, hidden_states_back = output_back[:3]
        #             output_embed_front = torch.squeeze(hidden_states_front[-1])
        #             output_embed_back = torch.squeeze(hidden_states_back[-1])
        #             output_embed = torch.cat([output_embed_front,output_embed_back],dim=0)
        #         else:
        #             outputs = self.model(input_ids, labels=None, output_hidden_states=True)
        #             _, hidden_states = outputs[:3]
        #             output_embed = torch.squeeze(hidden_states[-1])
        #         output_embeds.append(output_embed)
        # output_embeds = torch.stack(output_embeds)


        # 将所有的embeddings padding到最长的长度再用stack拼接
        # output_embeds = nn.utils.rnn.pad_sequence(output_embeds, batch_first=True)
        
        return output_embeds