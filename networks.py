import torch
import numpy as np

class DenseLayer(torch.nn.Module):
    def __init__(self,
                 c_in, # dimensionality of input features
                 c_out, # dimensionality of output features
                 zero_init=False, # initialize weights as zeros; use Xavier uniform init if zero_init=False
                 ):
        super().__init__()
        self.linear = torch.nn.Linear(c_in, c_out)
        if zero_init:
            torch.nn.init.zeros_(self.linear.weight.data)
        else:
            torch.nn.init.uniform_(self.linear.weight.data, -np.sqrt(6 / (c_in + c_out)), np.sqrt(6 / (c_in + c_out)))
        torch.nn.init.zeros_(self.linear.bias.data)
    def forward(self,node_feats):
        node_feats = self.linear(node_feats)
        return node_feats

class DeconvNet(torch.nn.Module):
    def __init__(self, gene_num, hidden_dims, n_celltypes):
        super(DeconvNet, self).__init__()
        self.hidden_dims = hidden_dims
        self.deconv_alpha_layer = DenseLayer(hidden_dims, 1, zero_init=True)
        self.deconv_beta_layer = DenseLayer(hidden_dims, n_celltypes, zero_init=True)
        self.gamma = torch.nn.Parameter(torch.Tensor(1, gene_num).zero_())

    def forward(self, z, count_matrix, library_size, basis):
        beta, alpha = self.deconv(z)
        log_lam = torch.log(torch.matmul(beta, basis) + 1e-6) + alpha + self.gamma
        lam = torch.exp(log_lam)

        decon_loss = -torch.mean(torch.sum(
                count_matrix * (torch.log(library_size + 1e-6) + log_lam) - library_size * lam, dim=1)
            )
        return decon_loss

    def deconv(self, z):
        beta = self.deconv_beta_layer(torch.nn.functional.elu(z))
        beta = torch.nn.functional.softmax(beta, dim=1)
        alpha = self.deconv_alpha_layer(torch.nn.functional.elu(z))
        return beta, alpha