#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from baseline_model import Generator, Discriminator
from baseline_config import *


def prepare_data(BATCH_SIZE):

    num_cores = os.cpu_count()

    transformation = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    train_set = torchvision.datasets.ImageFolder(
        '/data/cats',
        transform=transformation
    )

    dataset_size = len(train_set)
    trimmed_dataset_size = dataset_size - (dataset_size % BATCH_SIZE)
    train_set.samples = train_set.samples[:trimmed_dataset_size]
    train_set.targets = train_set.targets[:trimmed_dataset_size]

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_cores
        )

    return train_loader


def initialize_weights(mod):
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(mod.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(mod.weight.data, 1.0, 0.02)
        nn.init.constant_(mod.bias.data, 0)


def reset_gradients(network):
    for param in network.parameters():
        param.grad = None


def model_setup(device=DEVICE):
    generator = Generator()
    generator.apply(initialize_weights)
    generator.to(device)

    discriminator = Discriminator()
    discriminator.apply(initialize_weights)
    discriminator.to(device)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    return generator, discriminator, gen_optimizer, disc_optimizer

def train_setup(images, discriminator):

    actual_images = images.to(DEVICE)
    real_batch_size = discriminator(actual_images).view(-1).size(0)
    real_label = torch.ones((real_batch_size, ), dtype=torch.float32, device=DEVICE)
    fake_label = torch.zeros((real_batch_size, ), dtype=torch.float32, device=DEVICE)

    return actual_images, real_batch_size, real_label, fake_label