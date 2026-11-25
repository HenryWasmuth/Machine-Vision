
import torch
import torch.nn as nn
import csv
from pathlib import Path


RESULTS_DIR = Path("./results")
RESULTS_DIR.mkdir(exist_ok=True)


def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate_model(model, loader, criterion, device):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            predictions = outputs.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
            )

    accuracy = correct / total
    average_loss = total_loss / total

    return average_loss, accuracy, all_labels, all_predictions


def save_predictions(model_name, initialization,
                     labels, predictions):

    filename = (
        RESULTS_DIR /
        f"{model_name}_{initialization}_predictions.csv"
    )

    with open(filename, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(["true_label", "predicted_label"])

        for true, predicted in zip(labels, predictions):
            writer.writerow([true, predicted])


def train_and_test(model, model_name, initialization,
                   train_loader, val_loader, test_loader,
                   device, epochs=3):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    best_val_accuracy = 0
    best_epoch = 0
    best_state = None

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, _, _ = evaluate_model(
            model, val_loader, criterion, device
        )

        print(
            f"{model_name} | {initialization} | "
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_accuracy:

            best_val_accuracy = val_acc
            best_epoch = epoch + 1

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    # Load best model
    model.load_state_dict(best_state)

    # Save model weights
    checkpoint = (
        RESULTS_DIR /
        f"{model_name}_{initialization}.pth"
    )

    torch.save(model.state_dict(), checkpoint)

    # Final evaluation on test data
    test_loss, test_acc, labels, predictions = evaluate_model(
        model, test_loader, criterion, device
    )

    save_predictions(
        model_name, initialization, labels, predictions
    )

    return {
        "model": model_name,
        "initialization": initialization,
        "best_epoch": best_epoch,
        "validation_accuracy": best_val_accuracy,
        "test_accuracy": test_acc,
        "test_loss": test_loss
    }
