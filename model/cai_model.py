from utils.constant import pro2id
from utils.evaluate import get_cai_prob
import torch

class Cai_model():
    def __init__(self, cai_table):
        self.cai_prob_table = get_cai_prob(cai_table)
        self.pro2id = pro2id
        self.max_length = 1022
    def calculate_score(self, inputs, length):
        cai_list = []
        for i in range(len(inputs)):
            cai_score = [self.cai_prob_table[self.pro2id[inputs[i][step]]] for step in range(length[i])]
            cai_score = torch.stack(cai_score)
            if self.max_length-length[i]>0:
                cai_score = torch.cat([cai_score, torch.zeros(self.max_length-length[i], 6)])
            cai_list.append(cai_score)

        return torch.stack(cai_list)