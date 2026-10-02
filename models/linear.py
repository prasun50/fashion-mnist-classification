import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):

    def __init__(self, num_inputs=28*28, num_outputs=10):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.net = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, x):
        return self.net(x.reshape(x.shape[0], -1))

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def configure_optimizer(self, lr):
        return torch.optim.SGD(
            self.parameters(),
            lr=lr
        )

    def step(self, batch):
        return self.loss(self(batch[0]), batch[1])

    def training_step(self, batch):
        return self.step(batch)

    def validation_step(self, batch):
        return self.step(batch)

    def test_step(self, batch):
        return self.step(batch)