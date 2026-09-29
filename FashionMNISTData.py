import torch
import torchvision.transforms.v2 as v2
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt

class FashionMNISTData:

    def __init__(self, root="./data"):
        self.root = root
        self.transform = self.configure_transform()
        self._full_train_dataset = self.download_dataset(train=True)
        self.test_dataset = self.download_dataset(train=False)
        self.train_dataset, self.val_dataset = self.split_train_val(lengths=(50000, 10000))
        self.train_dataloader = self.create_dataloader(self.train_dataset)
        self.val_dataloader = self.create_dataloader(self.val_dataset)
        self.test_dataloader = self.create_dataloader(self.test_dataset)

        self.class_to_idx = self._full_train_dataset.class_to_idx
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def download_dataset(self, train):
        return FashionMNIST(
            root=self.root,
            train=train,
            download=True,
            transform=self.transform
        )

    def split_train_val(self, lengths):
        return torch.utils.data.random_split(self._full_train_dataset, lengths, generator=torch.Generator().manual_seed(5463))

    def create_dataloader(self, dataset, batch_size=32, shuffle=True):
        return torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle
        )

    def configure_transform(self):
        return v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

class FashionMNISTVisualizer:

    @staticmethod
    def visualize_samples(dataset, num_samples=5, label_to_class=None):
        fig, axes = plt.subplots(nrows=1, ncols=num_samples, figsize=(15, 2))
        for i in range(num_samples):
            image, label = dataset[i]
            axes[i].imshow(image.squeeze(0).numpy(), cmap='gray')
            if label_to_class:
                label = label_to_class[label]
            axes[i].set_title(label, fontsize=12, fontweight='bold')
            axes[i].axis('off')
        plt.show()

    @staticmethod
    def visualize_class(dataset, class_label, class_name, num_samples=5):
        class_indices = [label for label in range(len(dataset)) if dataset[label][1] == class_label]
        fig, axs = plt.subplots(nrows=1, ncols=num_samples, figsize=(15, 2))
        for i in range(num_samples):
            img, _ = dataset[class_indices[i]]
            axs[i].imshow(img.squeeze(0).numpy(), cmap="gray")
            axs[i].axis("off")
        fig.suptitle(class_name, fontsize=12, fontweight="bold")
        plt.show()

    @staticmethod
    def visualize_batch(dataloader, num_samples=5, label_to_class=None):
        batch = next(iter(dataloader))
        images, labels = batch
        fig, axes = plt.subplots(nrows=1, ncols=num_samples, figsize=(15, 2))
        for i in range(num_samples):
            img, label = images[i], labels[i].item()
            axes[i].imshow(images[i].squeeze(0).numpy(), cmap='gray')
            if label_to_class:
                label = label_to_class[label]
            axes[i].set_title(label, fontsize=12, fontweight='bold')
            axes[i].axis('off')
        plt.show()

if __name__ == "__main__":
    data = FashionMNISTData()
    print(f"Train dataset size: {len(data.train_dataset)}")
    print(f"Validation dataset size: {len(data.val_dataset)}")
    print(f"Test dataset size: {len(data.test_dataset)}")
    print(f"Train dataloader size: {len(data.train_dataloader)}")
    print(f"Validation dataloader size: {len(data.val_dataloader)}")
    print(f"Test dataloader size: {len(data.test_dataloader)}")

    batch = next(iter(data.train_dataloader))
    images, labels = batch
    print(f"Batch images shape: {images.shape}")
    print(f"Batch labels shape: {labels.shape}")
    print(f"Batch labels: {labels}")

    # Visualize samples from the training dataset
    FashionMNISTVisualizer.visualize_samples(data.train_dataset, label_to_class=data.idx_to_class)