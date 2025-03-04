import torch.nn as nn


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
