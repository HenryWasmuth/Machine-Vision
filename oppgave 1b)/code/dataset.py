import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet


# Settings
DATA_DIR = "./data"
BATCH_SIZE = 32
NUM_CLASSES = 37
SEED = 42


def get_dataloaders():

    # Preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Load training dataset
    full_train_dataset = OxfordIIITPet(
        root=DATA_DIR,
        split="trainval",
        target_types="category",
        transform=transform,
        download=True
    )

    # Load test dataset
    test_dataset = OxfordIIITPet(
        root=DATA_DIR,
        split="test",
        target_types="category",
        transform=transform,
        download=True
    )

    # 80/20 train-validation split
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    generator = torch.Generator().manual_seed(SEED)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader
