import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1_mean = nn.Linear(128 * 50 * 2, args.latent_dim)
        self.fc1_logvar = nn.Linear(128 * 50 * 2, args.latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        z_mean = self.fc1_mean(x)
        z_logvar = self.fc1_logvar(x)

        return z_mean, z_logvar

class Sampling(nn.Module):
    def __init__(self):
        super(Sampling, self).__init__()

    def forward(self, z_mean, z_logvar):
        batch_size, latent_dim = z_mean.size()
        epsilon = torch.randn(batch_size, latent_dim).to(z_mean.device)
        std = torch.exp(0.5 * z_logvar)
        z = z_mean + std * epsilon
        return z

class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(100 + args.latent_dim, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.relu = nn.ReLU()

    def forward(self, z, cond):
        x = torch.cat([z, cond.view(z.shape[0], -1)], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = x.view(x.shape[0], 1, 50, 2)
        return x

class CVAE(nn.Module):
    def __init__(self,args):
        super(CVAE, self).__init__()

        self.encoder = Encoder(args)
        self.sampling = Sampling()
        self.decoder = Decoder(args)

    def forward(self, x, cond):
        z_mean, z_logvar = self.encoder(x)
        z = self.sampling(z_mean, z_logvar)
        x_recon = self.decoder(z, cond)
        return x_recon, z_mean, z_logvar


# TODO - VUNet