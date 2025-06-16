from lstm.dataloader import H5Dataset
from lstm.model import DALSTMModel
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import random
import torch.optim as optim
from pathlib import Path
import pandas as pd


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalized_mae(y_true, y_pred):
    mae = np.average(np.abs(y_true - y_pred), axis=0)
    med = np.median(y_true)
    med_mae = np.average(np.abs(y_true - med), axis=0)
    return mae / med_mae


def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                scheduler,
                device,
                num_epochs,
                early_patience,
                min_delta,
                results_location
                ):

    if not Path(results_location).exists():
        Path(results_location).mkdir(parents=True, exist_ok=True)
    #Training loop
    current_patience = 0
    best_valid_loss = float('inf')
    best_nmae = float('inf')

    epochs = list()
    train_losses = list()
    valid_losses = list()
    train_nmaes = list()
    valid_nmaes = list()

    for epoch in range(num_epochs):
        # training
        model.train()
        epochs.append(epoch)
        mae = 0
        nmae = 0
        for batch in train_loader:
            # Forward pass
            inputs = batch[0].to(device)
            targets = batch[1].to(device)
            optimizer.zero_grad()  # Resets the gradients
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            mae += loss
            nmae += normalized_mae(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy())

        train_losses.append(mae.detach().cpu().numpy() / len(train_loader))
        train_nmaes.append(nmae / len(train_loader))
        model.eval()
        with torch.no_grad():
            total_valid_loss = 0
            total_nmae = 0
            for batch in val_loader:
                inputs = batch[0].to(device)
                targets = batch[1].to(device)
                outputs = model(inputs)
                valid_loss = criterion(outputs, targets)
                total_valid_loss += valid_loss.item()
                total_nmae += normalized_mae(targets.detach().cpu().numpy(), outputs.detach().cpu().numpy())
            average_valid_loss = total_valid_loss / len(val_loader)
            average_nmae = total_nmae / len(val_loader)
            valid_losses.append(average_valid_loss)
            valid_nmaes.append(average_nmae)
        # print the results
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {average_valid_loss}')

        new_best_mae = False
        new_best_nmae = False

        # save the best model based on mae
        if average_valid_loss < best_valid_loss - min_delta:
            best_valid_loss = average_valid_loss
            current_patience = 0
            new_best_mae = True
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
                'best_valid_loss': best_valid_loss,
                'best_nmae': best_nmae
            }
            checkpoint_path = results_location + '/best_mae_ckpt.pt'
            torch.save(checkpoint, checkpoint_path)

        if not new_best_mae and not new_best_nmae:
            current_patience += 1
            if current_patience >= early_patience:
                print('Early stopping: Val loss has not improved for {} epochs.'.format(early_patience))
                break

        # Update learning rate if there is any scheduler
        if scheduler is not None:
            scheduler.step(average_valid_loss)

        pd.DataFrame({'epoch': epochs, 'train_loss': train_losses, 'valid_loss': valid_losses, 'train_nmae': train_nmaes, 'valid_nmae': valid_nmaes}).to_csv(results_location + '/log.csv', index=False)

    return best_valid_loss, best_nmae


def set_optimizer (model, optimizer_type, base_lr, eps, weight_decay):
    if optimizer_type == 'NAdam':
        optimizer = optim.NAdam(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, eps=eps,
                                weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=base_lr, eps=eps,
                               weight_decay=weight_decay)
    else:
        raise ValueError('Invalid optimizer type. Must be one of: NAdam, AdamW, Adam')
    return optimizer


def train(batch_size,
          input_data_location,
          output_location,
          device,
          seed,
          max_training_epochs,
          n_neurons,
          n_layers,
          dropout,
          optimizer_type,
          base_lr,
          eps,
          weight_decay,
          early_stop_patience,
          early_stop_min_delta
          ):
    """
    Train LSTM model.

    Parameters
    ----------
    batch_size: int
    input_data_location: str
    output_location: str
    device: str
    seed: int
    max_training_epochs: int
    n_neurons: int
    n_layers: int
    dropout: float
    optimizer_type: str
    base_lr: float
    eps: float
    weight_decay: float
    early_stop_patience: int
    early_stop_min_delta: float

    Returns
    -------
    float
        Validation loss.
    """

    set_random_seed(seed)

    train_dataset = H5Dataset(input_data_location + "/train.hdf5",
                              x_name="train_X",
                              y_name="train_y")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    val_dataset = H5Dataset(input_data_location + "/train.hdf5",
                            x_name="validation_X",
                            y_name="validation_y")
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    # define loss function
    criterion = nn.L1Loss(reduction='mean')

    # define the model
    model = DALSTMModel(input_size=train_dataset.get_feature_dim(),
                        hidden_size=n_neurons,
                        n_layers=n_layers,
                        max_len=train_dataset.get_num_timesteps(),
                        dropout=dropout).to(device)
    # define optimizer
    optimizer = set_optimizer(model,
                              optimizer_type,
                              base_lr,
                              eps,
                              weight_decay)
    # define scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    val_loss = train_model(model=model,
                           train_loader=train_loader,
                           val_loader=val_loader,
                           criterion=criterion,
                           optimizer=optimizer,
                           scheduler=scheduler,
                           device=device,
                           num_epochs=max_training_epochs,
                           early_patience=early_stop_patience,
                           min_delta=early_stop_min_delta,
                           results_location=output_location)

    return val_loss
