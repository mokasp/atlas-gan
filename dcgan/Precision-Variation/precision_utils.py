#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from precision_model import Generator, Discriminator
from precision_config import *


def prepare_data(BATCH_SIZE):

    num_cores = os.cpu_count()

    transformation = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.1307,), std=(0.3081,))
    ])
    train_set = torchvision.datasets.MNIST(
        'content/train',
        train=True,
        download = True,
        transform=transformation
    )

    dataset_size = len(train_set)
    trimmed_dataset_size = dataset_size - (dataset_size % BATCH_SIZE)
    train_set.data = train_set.data[:trimmed_dataset_size]
    train_set.targets = train_set.targets[:trimmed_dataset_size]

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_cores
        )

    return train_loader


def initialize_weights(mod):
    if isinstance(mod, nn.Conv2d) or isinstance(mod, nn.ConvTranspose2d):
        nn.init.normal_(mod.weight, 0.0, 0.02)
    elif isinstance(mod, nn.BatchNorm2d):
        nn.init.normal_(mod.weight, 1.0, 0.02)
        nn.init.constant_(mod.bias, 0)


def reset_gradients(network):
    for param in network.parameters():
        param.grad = None


def model_setup(device=DEVICE):
    generator = Generator()
    generator.apply(initialize_weights)
    generator.to(device).double()

    discriminator = Discriminator()
    discriminator.apply(initialize_weights)
    discriminator.to(device).double()

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=GEN_LEARNING_RATE, betas=(BETA1, BETA2))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=DISC_LEARNING_RATE, betas=(BETA1, BETA2))

    calc_loss = nn.BCELoss()

    return generator, discriminator, gen_optimizer, disc_optimizer, calc_loss

def train_setup(images, discriminator):

    actual_images = images.to(DEVICE).double()
    real_batch_size = discriminator(actual_images).view(-1).size(0)
    real_label = torch.ones((real_batch_size, ), dtype=torch.float64, device=DEVICE)
    fake_label = torch.zeros((real_batch_size, ), dtype=torch.float64, device=DEVICE)

    return actual_images, real_batch_size, real_label, fake_label