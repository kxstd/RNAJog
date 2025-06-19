from model.actor import Actor
from model.critic import Critic
from model.embedder_pretrain import Embedder
import torch.nn as nn
import torch
from utils.random_gen import codon_gen, cod2case_mul, cod2case
from utils.constant import get_mask_f, trans_idx2codon
from utils.evaluate import calculate_mfe_mul


class Actor_Critic(nn.Module):
    def __init__(self,cfg):
        super(Actor_Critic,self).__init__()
        self.seed = cfg.seed
        self.device = torch.device(cfg.device)

        self.embedder = Embedder(cfg)
        self.actor = Actor(cfg)
        self.critic = Critic(cfg)
        self.get_mask = get_mask_f()

        self.to(self.device) 

    def forward(self, codons, pro, length, merge_prob, alpha):
        embeddings = self.embedder(codons)
        outputs, log_probs = self.actor(embeddings, pro, length, merge_prob, alpha)
        b = self.critic(embeddings)

        return outputs, log_probs, b
    
    def optimize(self, codons, pro, length, merge_prob, alpha, sample_method, sample_temperature, ban_codon_table, ban_pro_seqs):
        if ban_pro_seqs:
            for ban_pro_seq in ban_pro_seqs:
                for pro_seq in pro:
                    if ban_pro_seq in "".join(pro_seq):
                        raise ValueError("Protein sequence {} is banned".format(ban_pro_seq))
        embeddings = self.embedder(codons)
        outputs, log_probs = self.actor.optimize(embeddings, pro, length, merge_prob, alpha, sample_method, sample_temperature, ban_codon_table)

        return outputs, log_probs
