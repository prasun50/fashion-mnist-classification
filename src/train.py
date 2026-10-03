import argparse
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import torch
import utils

# path to models directory to import from models/
path_to_models = Path(__file__).parent.parent / "models"
if str(path_to_models) not in sys.path:
    sys.path.append(str(path_to_models))

# import modules located inside models/
import linear

class Trainer:

    def __init__(self, batch_size=32, max_epoches=1, lr=1e-3):
        self.batch_size = batch_size
        self.max_epoches = max_epoches
        self.lr = lr

    def fit(self, model, data):
        print(f"fitting {model.__class__.__name__}...")
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
        self.train_dataset, self.val_dataset = utils.split_dataset(
            dataset = self.data.full_train_dataset,
            lengths = [len(self.data.full_train_dataset)-len(self.data.test_dataset), len(self.data.test_dataset)]
        )
        self.test_dataset = self.data.test_dataset
        self.train_dataloader = utils.get_dataloader(self.train_dataset, batch_size=self.batch_size)

    def calculate_accuracy(self):
        loader = utils.get_dataloader(self.val_dataset, batch_size=len(self.val_dataset))
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
        loader = utils.get_dataloader(self.val_dataset, batch_size=len(self.val_dataset))
        batch = next(iter(loader))
        self.model.eval()
        with torch.no_grad():
            loss = self.model.validation_step(batch).item()
        self.model.train()
        return loss

    def visualize_wrong_predictions(self):
        loader = utils.get_dataloader(self.val_dataset)
        x, y = next(iter(loader))
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

    def plot_training_loss(self):
        losses = self.training_losses.reshape(-1)
        losses = losses[: losses.shape[0] - losses.shape[0] % 100]
        losses = losses.reshape(-1, 100).mean(dim=1)

        fig, ax = plt.subplots(layout="constrained")
        ax.plot(losses.numpy())
        ax.set_xlabel("batch number")
        ax.set_ylabel("loss")
        ax.set_title("Training Loss", fontsize=14, fontweight="bold")
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train model on FashionMNIST data")
    parser.add_argument("--max-epoches", type=int, default=1, help="number of epoches")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--model", default="LinearModel", choices=["LinearModel"], help="model to train the data on")
    parser.add_argument("-v", "--visualize-wrong-preds", action="store_true", help="visualize wrong predictions")
    args = parser.parse_args()
    
    data = utils.FashionMNISTData()
    model = getattr(linear, args.model)()
    trainer = Trainer(batch_size=args.batch_size, max_epoches=args.max_epoches, lr=args.learning_rate)
    trainer.fit(model, data)

    if args.visualize_wrong_preds:
        trainer.visualize_wrong_predictions()