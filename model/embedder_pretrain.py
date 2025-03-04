import torch
import torch.nn as nn
from transformers import BertForSequenceClassification
from utils.tokenizer import get_tokenizer

class Embedder(nn.Module):
    def __init__(self,cfg):
        super(Embedder,self).__init__()
        self.tokenizer = get_tokenizer()
        self.model = BertForSequenceClassification.from_pretrained(cfg.embedder.model_dir)
        self.model.eval()
        self.cuda = (cfg.device == "cuda")
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

        return output_embeds