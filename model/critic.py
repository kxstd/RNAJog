import torch.nn as nn
import torch
import math


class Critic(nn.Module):
    def __init__(self, cfg):
        super(Critic, self).__init__()
        embedding_size = cfg.embedder.embedding_size
        hidden_size = cfg.critic.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

    def forward(self, encodings):
        baseline_prediction = self.mlp(encodings[:,0,:])

        return baseline_prediction.squeeze(-1)

# class Critic(nn.Module):
#     def __init__(self, cfg):
#         super(Critic, self).__init__()
#         #  embedding_size, hidden_size, num_layers, glimpse_size, P, 
#         embedding_size = cfg.embedder.embedding_size
#         glimpse_size = cfg.critic.glimpse_size
#         self.glimpse = Glimpse(embedding_size,glimpse_size)
#         self.decoder = nn.Sequential(
#             nn.Linear(embedding_size, glimpse_size),
#             nn.ReLU(),
#             nn.Linear(glimpse_size, 1),
#             nn.ReLU()
#         )

#     def forward(self, encodings):
#         baseline_prediction = self.decoder(encodings)

#         return baseline_prediction.squeeze()


# class Glimpse(nn.Module):
#     def __init__(self, d_model, d_unit):
#         super(Glimpse, self).__init__()
#         self.tanh = nn.Tanh()
#         self.conv1d = nn.Conv1d(d_model, d_unit, 1)
#         self.v = nn.Parameter(torch.FloatTensor(d_unit), requires_grad=True)
#         self.v.data.uniform_(-(1. / math.sqrt(d_unit)), 1. / math.sqrt(d_unit))
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, encs):
#         # encs: batch * length * d_model
#         encoded = self.conv1d(encs.permute(0, 2, 1)).permute(0, 2, 1)
#         scores = torch.sum(self.v * self.tanh(encoded), -1)
#         attention = self.softmax(scores)
#         glimpse = attention.unsqueeze(-1) * encs
#         glimpse = torch.sum(glimpse, 1)
#         print(glimpse.shape)
#         return glimpse