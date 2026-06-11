import torch.nn as nn
import torch


class Papyrus2ndStage(nn.Module):
    def __init__(self):
        super().__init__()

        Nzernike = 50

        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=11, padding=7),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=7, padding=5),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, padding=3),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=2, padding=2),
            nn.LeakyReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.1),
        )

        self.outputlayer = nn.Sequential(
            nn.Linear(512, Nzernike),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.outputlayer(x)
        return x