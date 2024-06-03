import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MaskFuncGlobal(nn.Module):
    def __init__(self, args):
        super(MaskFuncGlobal, self).__init__()
        self.args = args
        self.state_dim = int(np.prod(args.state_shape))
        state_mask_dim = args.state_mask_dim

        self.fc1 = nn.Linear(self.state_dim, state_mask_dim)
        self.fc2 = nn.Linear(state_mask_dim, state_mask_dim)
        self.fc3 = nn.Linear(state_mask_dim, self.state_dim)

    def agent_init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent_init_hidden().unsqueeze(0).expand(batch_size, self.args.n_agents, -1)  # bav

    def forward(self, inputs):

        x = F.relu(self.fc1(inputs))
        x1 = F.relu(self.fc2(x))
        x2 = F.sigmoid(self.fc3(x1))
        return x2



