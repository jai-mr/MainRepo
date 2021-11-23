import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from utils import helper


def show_batch(data_loader, labels):
    """
    Visualize a batch of images.
    """
    images, targets = next(iter(data_loader))
    plt.figure(figsize=(16, 8))
    for i in range(28):
        ax = plt.subplot(4, 7, i + 1)
        ax.imshow(images[i].permute(1, 2, 0))
        plt.title(labels[targets[i]])
        plt.axis("off")


def plot_metrics(exp_metrics):
    """
    Plot Train and Test Accuracy and Loss.
    """
    sns.set(font_scale=1)
    plt.rcParams["figure.figsize"] = (25, 6)
    train_accuracy, train_losses, test_accuracy, test_losses = exp_metrics

    # Plot the learning curve.
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.array(train_losses), "b", label="Train Loss")

    # Label the plot.
    ax1.set_title("Train Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(np.array(train_accuracy), "b", label="Train Accuracy")

    # Label the plot.
    ax2.set_title("Train Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.show()

    # Plot the learning curve.
    _, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(np.array(test_losses), "b", label="Test Loss")

    # Label the plot.
    ax1.set_title("Test Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(np.array(test_accuracy), "b", label="Test Accuracy")

    # Label the plot.
    ax2.set_title("Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()

    plt.show()


def misclassified_images(model, test_loader, device, class_names=None, n_images=20):
    """
    Get misclassified images.
    """
    wrong_images = []
    wrong_label = []
    correct_label = []
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True).squeeze()  # get the index of the max log-probability

            wrong_pred = pred.eq(target.view_as(pred)) == False
            wrong_images.append(data[wrong_pred])
            wrong_label.append(pred[wrong_pred])
            correct_label.append(target.view_as(pred)[wrong_pred])

            wrong_predictions = list(zip(torch.cat(wrong_images), torch.cat(wrong_label), torch.cat(correct_label)))
        print(f"Total wrong predictions are {len(wrong_predictions)}")

        plot_misclassified_images(wrong_predictions, class_names=class_names, n_images=n_images)

    return wrong_predictions


def plot_misclassified_images(wrong_predictions, n_images=20, class_names=None):
    """
    Plot the misclassified images.
    """
    if class_names is None:
        class_names = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig = plt.figure(figsize=(10, 12))
    fig.tight_layout()
    mean, std = helper.calculate_mean_std("TinyImageNet")
    for i, (img, pred, correct) in enumerate(wrong_predictions[:n_images]):
        img, pred, target = img.cpu().numpy().astype(dtype=np.float32), pred.cpu(), correct.cpu()
        for j in range(img.shape[0]):
            img[j] = (img[j] * std[j]) + mean[j]

        img = np.transpose(img, (1, 2, 0))
        ax = fig.add_subplot(5, 5, i + 1)
        ax.axis("off")
        ax.set_title(f"\nactual : {class_names[target.item()]}\npredicted : {class_names[pred.item()]}", fontsize=10)
        ax.imshow(img)

    plt.show()
