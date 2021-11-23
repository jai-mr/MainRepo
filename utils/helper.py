import random

import numpy as np
import torch
import torchvision.transforms as T
from torchsummary import summary
from torchvision import datasets


def model_summary(model, input_size):
    """
    Summary of the model.
    """
    summary(model, input_size=input_size)


def get_default_device():
    """
    Pick GPU if available, else CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def seed_everything(seed: int):
    """
    Seed everything for reproducibility and deterministic behaviour.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def class_level_accuracy(model, loader, device, classes):
    """
    Accuracy per class level.
    """
    class_correct = list(0.0 for i in range(len(classes)))
    class_total = list(0.0 for i in range(len(classes)))

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader, 0):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct = (predicted == labels).squeeze()

            for i, label in enumerate(labels):
                class_correct[label] += correct[i].item()
                class_total[label] += 1

    for i in range(len(classes)):
        print("Accuracy of %5s : %2d %%" % (classes[i], 100 * class_correct[i] / class_total[i]))


def unnormalize(img):
    """
    De-normalize the image.
    """
    mean = (0.49139968, 0.48215841, 0.44653091)
    std = (0.24703223, 0.24348513, 0.26158784)
    img = img.cpu().numpy().astype(dtype=np.float32)

    for i in range(img.shape[0]):
        img[i] = (img[i] * std[i]) + mean[i]

    return np.transpose(img, (1, 2, 0))


def calculate_mean_std(dataset):
    """
    Calculate mean and std for a dataset.
    """
    if dataset == "CIFAR10":
        train_transform = T.ToTensor()
        train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        mean = train_set.data.mean(axis=(0, 1, 2)) / 255
        std = train_set.data.std(axis=(0, 1, 2)) / 255
        return mean, std

    if dataset == "TinyImageNet":
        return ((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
