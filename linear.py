import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import FashionMNISTData

# just a linear layer
class LinearModel(nn.Module):

    def __init__(self, num_inputs, num_outputs):
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

class Traniner:

    def __init__(self, max_epoches=1, lr=1e-3):
        self.max_epoches = max_epoches
        self.lr = lr

    def fit(self, model, data):
        print("fitting model...")
        self.configure_model(model)
        self.configure_data(data)
        self.epoch = 0
        self.training_losses = torch.empty((self.max_epoches, len(self.train_dataloader)), dtype=torch.float32)
        for self.epoch in range(self.max_epoches):
            self.fit_epoch()
            train_loss = self.training_losses[self.epoch].mean().item()
            val_loss = self.calculate_val_loss()
            accuracy = self.calculate_accuracy()
            print(f"epoch: {self.epoch}, train loss: {train_loss:.2f}, val loss: {val_loss:.2f}, accuracy: {accuracy:.2f}%")

    def fit_epoch(self):
        self.model.train()
        for i, batch in enumerate(self.train_dataloader):
            loss = self.model.training_step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.training_losses[self.epoch, i] = loss.item()

    def configure_model(self, model):
        self.model = model
        model.trainer = self
        self.optimizer = model.configure_optimizer(self.lr)

    def configure_data(self, data):
        self.data = data
        self.train_dataloader = data.train_dataloader
        self.val_dataloader = data.val_dataloader
        self.test_dataloader = data.test_dataloader

    def calculate_accuracy(self):
        loader = self.data.create_dataloader(self.data.val_dataset, batch_size=len(self.data.val_dataset))
        x, y = next(iter(loader))
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(x)
        self.model.train()
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
        preds = probs.max(dim=1).indices
        accuracy = (preds == y).sum() / len(preds)
        return accuracy * 100

    def calculate_val_loss(self):
        loader = self.data.create_dataloader(self.data.val_dataset, batch_size=len(self.data.val_dataset))
        batch = next(iter(loader))
        self.model.eval()
        with torch.no_grad():
            loss = self.model.training_step(batch).item()
        self.model.train()
        return loss

    def visualize_wrong_predictions(self):
        x, y = next(iter(self.val_dataloader))
        self.model.eval()
        with torch.inference_mode():
            logits = self.model(x)
        self.model.train()
        logits = logits - logits.max(dim=1, keepdim=True).values
        probs = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
        preds = probs.max(dim=1).indices

        incorrect = preds != y
        incorrect_indices, = incorrect.nonzero(as_tuple=True)
        incorrect_preds = preds[incorrect]
        correct_ys = y[incorrect]

        fig, axs = plt.subplots(ncols=len(incorrect_indices), figsize=(15, 3), layout="constrained")
        for i in range(len(incorrect_indices)):
            img, incorrect_label, actual_label = x[incorrect_indices[i]], incorrect_preds[i].item(), correct_ys[i].item()
            incorrect_cls, actual_cls = self.data.idx_to_class[incorrect_label], self.data.idx_to_class[actual_label]
            axs[i].imshow(img.squeeze(0), cmap="gray")
            axs[i].axis("off")
            axs[i].set_title(f"act: {actual_cls}\npred: {incorrect_cls}")
        fig.suptitle("Actual Class and Predicted Class", fontsize=14, fontweight="bold")
        plt.show()

if __name__ == "__main__":
    data = FashionMNISTData.FashionMNISTData()
    model = LinearModel(28*28, len(data.class_to_idx))
    trainer = Traniner(3, lr=0.1)
    trainer.fit(model, data)
    trainer.visualize_wrong_predictions()