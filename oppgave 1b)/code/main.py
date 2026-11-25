
import torch
import csv
from pathlib import Path

from dataset import get_dataloaders
from models import get_model
from evaluate import train_and_test


SEED = 42
EPOCHS = 3

MODEL_NAMES = [
    "resnet18",
    "mobilenet_v3",
    "efficientnet_b0"
]

INITIALIZATIONS = [
    True,   # Pretrained
    False   # Random
]


def main():

    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    # Load datasets
    train_loader, val_loader, test_loader = get_dataloaders()

    results = []

    # Run all experiments
    for model_name in MODEL_NAMES:

        for pretrained in INITIALIZATIONS:

            torch.manual_seed(SEED)

            initialization = (
                "pretrained" if pretrained else "random"
            )

            print(
                f"\nStarting {model_name} - {initialization}"
            )

            model = get_model(
                model_name,
                pretrained=pretrained
            )

            model = model.to(device)

            result = train_and_test(
                model=model,
                model_name=model_name,
                initialization=initialization,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                device=device,
                epochs=EPOCHS
            )

            results.append(result)

    # Save experiment results
    filename = Path("./results/baseline_results.csv")

    with open(filename, "w", newline="") as file:

        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys()
        )

        writer.writeheader()
        writer.writerows(results)

    print("\nAll experiments completed!")
    print("Results saved to:", filename)


if __name__ == "__main__":
    main()
