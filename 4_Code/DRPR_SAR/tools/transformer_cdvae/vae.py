import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class TransformerCVAE(nn.Module):
    def __init__(self, latent_dim=768, base_dim=64, in_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.base_dim = base_dim
        self.feature_shape = (base_dim * 4, 28, 28)
        feature_dim = self.feature_shape[0] * self.feature_shape[1] * self.feature_shape[2]

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim, base_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_dim * 2, base_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            ResBlock(base_dim * 4),
            ResBlock(base_dim * 4),
        )

        self.fc_mu = nn.Linear(feature_dim, latent_dim)
        self.fc_logvar = nn.Linear(feature_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, feature_dim)

        self.decoder = nn.Sequential(
            ResBlock(base_dim * 4),
            nn.ConvTranspose2d(base_dim * 4, base_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_dim * 2, base_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_dim, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        h_flat = h.flatten(1)
        return h, self.fc_mu(h_flat), self.fc_logvar(h_flat)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z):
        h = self.fc_decode(z).view(-1, *self.feature_shape)
        return self.decoder(h)

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        gx = self.decode(z)
        return gx, z, mu, logvar

