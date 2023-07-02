import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, intermediate_dim, device, num_hidden_layers = 1):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        if type(intermediate_dim) == type([]):
            self.num_hidden_layers = len(intermediate_dim)
            self.input_layer = nn.Sequential(nn.Linear(in_dim, intermediate_dim[0], device=device), nn.LeakyReLU())
            self.output_layer = nn.Sequential(nn.Linear(intermediate_dim[-1], out_dim, device=device))
            list_intermediate = [nn.Sequential(nn.Linear(intermediate_dim[i], intermediate_dim[i+1], device=device), nn.LeakyReLU())
                             for i in range(self.num_hidden_layers-1)]
            self.linear_relu_stack = nn.Sequential(*[layer for layer in list_intermediate])
        else:
            self.input_layer = nn.Sequential(nn.Linear(in_dim, intermediate_dim, device=device), nn.LeakyReLU())
            self.output_layer = nn.Sequential(nn.Linear(intermediate_dim, out_dim, device=device))
            list_intermediate = [nn.Sequential(nn.Linear(intermediate_dim, intermediate_dim, device=device), nn.LeakyReLU())
                                 for _ in range(num_hidden_layers)]
            self.linear_relu_stack = nn.Sequential(*[layer for layer in list_intermediate])

    def forward(self, x):
        #x = self.flatten(x)
        x = self.input_layer(x)
        hidden = self.linear_relu_stack(x)
        logits = self.output_layer(hidden)
        return logits
