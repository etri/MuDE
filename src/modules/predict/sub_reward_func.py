import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SubRewardFunc(nn.Module):
    #def __init__(self, input_shape, args):
    def __init__(self, scheme, args):
        super(SubRewardFunc, self).__init__()
        
        self.args = args
        self.state_dim = int(np.prod(args.state_shape))
        subRewardDim = 256
        actionDim = 16
        input_shape = self.state_dim  + actionDim        
        n_out = 1
        self.flag = 1

        self.fc1 = nn.Linear(input_shape, subRewardDim)
        self.fc2 = nn.Linear(subRewardDim, subRewardDim)
        self.fc3 = nn.Linear(subRewardDim, n_out)

        self.fc4 = nn.Linear(args.n_agents, actionDim)
        self.fc5 = nn.Linear(actionDim, actionDim)

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, ep_batch, mask):

        states = ep_batch["state"][:, :-1]
        masking_s = states * mask[:, :-1]
        actions = ep_batch["actions"][:,:-1]
        encode_actions1 = self.fc4(actions.squeeze(dim=3).float())
        encode_actions = self.fc5(encode_actions1)
        inputs = th.cat([masking_s, encode_actions], dim=2)
        x = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(x))
        if self.flag:
            q = F.relu(self.fc3(h))
            #q = th.abs(self.fc3(h))
        else:
            q = -F.relu(self.fc3(h))
            #q = -th.abs(self.fc3(h))
        return q




