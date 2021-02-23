import torch
import torch.nn as nn

import torch.optim as optim

TT = torch.TensorType

class AE(nn.Module):
    """Autoencoder"""

    def __init__(self, idim: int, hdim: int, odim: int):
        super(AE, self).__init__()
        self.enc_input = nn.Linear(idim, hdim)
        self.enc_output = nn.Linear(hdim, odim)
        self.dec_input = nn.Linear(odim, hdim)
        self.dec_output = nn.Linear(hdim, idim)
        self.activ = nn.ReLU()

    def encoder(self, X: TT) -> TT:
        # returns the code that will be used in clustering after training
        encoded = self.enc_input.forward(X)
        activated = self.activ(encoded)
        encoded = self.enc_output.forward(activated)
        code = self.activ(encoded)
        return code

    def decoder(self, X: TT) -> TT:
        decoded = self.dec_input.forward(X)
        activated = self.activ(decoded)
        decoded = self.dec_output.forward(activated)
        reconstructed = self.activ(decoded)
        return reconstructed

    def forward(self, X: TT) -> TT:
        # returns reconstruction of the initial vector for the training
        code = self.encoder(X)
        reconstructed = self.decoder(code)
        return reconstructed