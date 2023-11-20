import torch
from torch.utils.data import DataLoader
import numpy as np
from dataclasses import dataclass
from rich.progress import track
import optuna

from utils import SignalDataset, loss_function, set_seed
from model import CVAE


@dataclass
class Args():
    # training arguments
    batch_size: int = 250
    num_epochs: int = 20
    lr: float = 0.001
    seed: int = 1234
    # model arguments
    latent_dim: int = 8
    # loss arguments
    beta: float = 0.1
    wx: float = 0.1
    wy: float = 0.1
args = Args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(args.seed)

# Set functions to reconstruct
def f_y_0(x0, x1):
    return np.cos(49 * x0 + 42 * x1)

def f_y_1(x0, x1):
    return np.cos(56 * x0 + 63 * x1)

functions = [f_y_0, f_y_1]

# Load datasets
train_ds = SignalDataset(torch.load('train_dataset.pt'))
val_ds = SignalDataset(torch.load('validation_dataset.pt'))
scoring_ds = SignalDataset(torch.load('scoring_dataset.pt'))

train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=args.batch_size)
scoring_dl = DataLoader(scoring_ds, batch_size=len(scoring_ds))

# Define model
model = CVAE(args)
model.to(device)

# Define optimizer and loss
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# Train model
model.train()

for epoch in range(args.num_epochs):
    train_loss = 0.0
    KLD_loss = 0.0
    recon_loss = 0.0
    cond_loss = 0.0

    for batch_idx, (data, condition) in enumerate(track(train_dl)):
        data.to(device)
        condition.to(device)

        recon_data, z_mean, z_logvar = model(data, condition)
        loss, loss_KDL, loss_x, loss_y = loss_function(recon_data, data, condition, z_mean, z_logvar, args.beta, args.wx, args.wy, functions)
        train_loss += loss.item()
        KLD_loss += loss_KDL.item()
        recon_loss += loss_x.item()
        cond_loss += loss_y.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dl)
    KLD_loss /= len(train_dl)
    recon_loss /= len(train_dl)
    cond_loss /= len(train_dl)
    print('Train Epoch {}: Average Loss: {:.6f}, KDL: {:3f}, x_loss: {:3f}, y_loss: {:3f}'.format(epoch, train_loss, KLD_loss, recon_loss, cond_loss))

# Evaluate model
model.eval()

val_loss = 0.0
with torch.no_grad():
    for batch_idx, (data, condition) in enumerate(track(val_dl)):
        data.to(device)
        condition.to(device)

        recon_data, z_mean, z_logvar = model(data, condition)
        loss,_,_,_ = loss_function(recon_data, data, condition, z_mean, z_logvar, 0.1, 0.1, 0.1, functions)
        val_loss += loss.item()

    val_loss /= len(val_dl)
    print('Test Loss: {:.6f}'.format(val_loss))

# Sample and save prediction for submission
model.eval()

num_samples = 30
x_batch, y_batch = next(iter(scoring_dl))
x_outputs, y_outputs = [], []
for idx in range(len(y_batch)):
    given_y = y_batch[idx].unsqueeze(0).to(device)
    given_y = torch.reshape(given_y, (len(given_y), 100))

    samples = []
    givens = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(1, args.latent_dim).to(device)

            sample = model.decoder(z, given_y)
            samples.append(sample)
            givens.append(given_y)

    x_output = torch.cat(samples, dim=0)
    y_output = torch.cat(givens, dim=0)

    x_outputs.append(x_output)
    y_outputs.append(y_output.reshape(x_output.shape))

x_outputs = torch.cat(x_outputs, dim=0)
y_outputs = torch.cat(y_outputs, dim=0)

ds = torch.utils.data.TensorDataset(x_outputs,y_outputs)
torch.save(ds, 'result_dataset.pt')
