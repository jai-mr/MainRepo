import torch.nn as nn
from tqdm import tqdm


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train(
    model,
    device,
    train_loader,
    optimizer,
    train_acc,
    train_loss,
    lambda_l1,
    criterion,
    lrs,
    scheduler,
    grad_clip=None,
):
    """
    Train the model.
    """
    model.train()
    pbar = tqdm(train_loader)

    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        y_pred = model(data)

        loss = criterion(y_pred, target)

        # L1 Regularization
        if lambda_l1 > 0:
            l1 = 0
            for p in model.parameters():
                l1 = l1 + p.abs().sum()
                loss = loss + lambda_l1 * l1

        train_loss.append(loss.data.cpu().numpy().item())

        # Backpropagation
        loss.backward()

        # Gradient clipping
        if grad_clip:
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)

        optimizer.step()

        if scheduler:
            scheduler.step()

        lrs.append(get_lr(optimizer))

        # Update pbar-tqdm

        pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"Train Loss={loss.item()} Batch_id={batch_idx} LR={lrs[-1]: 0.5f} Train Accuracy={100 * correct / processed: 0.2f}"
        )
        train_acc.append(100 * correct / processed)
