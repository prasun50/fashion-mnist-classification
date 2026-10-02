from pathlib import Path
import torch
import torchvision.transforms.v2 as v2
from torchvision.datasets import FashionMNIST
import matplotlib.pyplot as plt

data_root = Path(__file__).parent.parent / "data"
default_transform = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(dtype=torch.float32, scale=True)
])

def get_dataset(train=True, transform=default_transform):
    return FashionMNIST(
        root=data_root,
        train=train,
        transform=transform,
        download=True
    )

def split_dataset(dataset, lengths, deterministic=False, seed=43567):
    gen = torch.Generator()
    if deterministic:
        gen = gen.manual_seed(seed)
    return torch.utils.data.dataset.random_split(
        dataset=dataset,
        lengths=lengths,
        generator=gen
    )

def get_dataloader(dataset, batch_size=32, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

def draw_image(img, label=None):
    _, ax = plt.subplots(layout="constrained", figsize=(4, 4))
    ax.imshow(img.squeeze().numpy(), cmap="gray")
    ax.axis("off")
    if label is not None:
        ax.set_title(label, fontsize=14, fontweight="bold")
    plt.show()

def draw_samples(imgs, labels):
    assert len(imgs) == len(labels) or len(labels) == 1, "imgs and lables must be of same length"
    n = len(imgs)
    fig, axs = plt.subplots(ncols=n, figsize=(15, 3), layout="constrained")
    for i in range(n):
        axs[i].imshow(imgs[i].squeeze().numpy(), cmap="gray")
        axs[i].axis("off")
        if len(labels) != 1:
            axs[i].set_title(labels[i], fontsize=14, fontweight="bold")
    if len(labels) == 1:
        fig.suptitle(*labels, fontsize=14, fontweight="bold")
    plt.show()

class FashionMNISTData:

    def __init__(self, transform=default_transform):
        self.batch_size = 32
        self.full_train_dataset = get_dataset(train=True, transform=transform)
        self.test_dataset = get_dataset(train=False, transform=transform)
        self.class_to_idx = self.full_train_dataset.class_to_idx
        self.idx_to_class = {idx: cl for cl, idx in self.class_to_idx.items()}

if __name__ == "__main__":
    mnist_data = FashionMNISTData()
    idx_to_class = mnist_data.idx_to_class

    imgs, labels = [], []
    for i in range(8):
        img, label = mnist_data.full_train_dataset[i]
        label = idx_to_class[label]
        imgs.append(img); labels.append(label)
    draw_samples(imgs, labels)