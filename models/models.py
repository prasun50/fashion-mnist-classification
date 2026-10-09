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

class LeNet(nn.Module):

    def __init__(self, num_outputs=10):
        super().__init__()
        self.num_outputs = num_outputs
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=400, out_features=120), nn.ReLU(),
            nn.Linear(in_features=120, out_features=84), nn.ReLU(),
            nn.Linear(in_features=84, out_features=self.num_outputs)
        )
        self.apply(self.apply_init)

    def apply_init(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        return self.net(x)

    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def step(self, batch):
        return self.loss(self(batch[0]), batch[1])

    def training_step(self, batch):
        return self.step(batch)

    def validation_step(self, batch):
        return self.step(batch)

    def configure_optimizer(self, lr):
        return torch.optim.SGD(
            params=self.parameters(),
            lr=lr
        )
    
    def layer_summary(self, x_shape):
        x = torch.normal(0.0, 1.0, size=x_shape)
        for layer in self.net:
            x = layer(x)
            print(f"{layer.__class__.__name__} output shape:\t{x.shape}")