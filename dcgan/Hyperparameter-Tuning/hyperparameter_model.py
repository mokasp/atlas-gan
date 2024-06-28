# specific functions related to neural networks
import torch.nn as nn

from hyperparameters_config import *


class Generator(nn.Module):

  def __init__(self):
    super(Generator, self).__init__()

    # set up model architecture
    # 4 convoloutions total
    # input batch_size1x1x1 vector
    # output batch_size1x28x28
    self.network = nn.Sequential(
        nn.ConvTranspose2d(
            in_channels=LATENT_DIMS,
            out_channels=NUM_GEN_KERN * 4  ,
            kernel_size=4,
            stride=2,
            padding=0,
            bias=False
            ),
        nn.BatchNorm2d(NUM_GEN_KERN * 4),
        nn.ReLU(True),
        nn.ConvTranspose2d(
            in_channels=NUM_GEN_KERN * 4,
            out_channels=NUM_GEN_KERN * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
            ),
        nn.BatchNorm2d(NUM_GEN_KERN * 2),
        nn.ReLU(True),
        nn.ConvTranspose2d(
            in_channels=NUM_GEN_KERN * 2,
            out_channels=NUM_GEN_KERN,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
            ),
        nn.BatchNorm2d(NUM_GEN_KERN),
        nn.ReLU(True),
        nn.ConvTranspose2d(
            in_channels=NUM_GEN_KERN,
            out_channels=NUM_CHANNELS,
            kernel_size=4,
            stride=2,
            padding=3,
            bias=False
            ),
        nn.Tanh(),
    )

  def forward(self, input):
    return self.network(input)



class Discriminator(nn.Module):

  def __init__(self):
    super(Discriminator, self).__init__()

    # set up model architecture
    # 4 convoloutions total
    # input batch_size1x28x28
    # output batch_sizex1x1x1
    self.network = nn.Sequential(
        nn.Conv2d(
            in_channels=NUM_CHANNELS,
            out_channels=NUM_DISC_KERN,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
            ),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(
            in_channels=NUM_DISC_KERN,
            out_channels=NUM_DISC_KERN * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
            ),
        nn.BatchNorm2d(NUM_DISC_KERN * 2),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(
            in_channels=NUM_DISC_KERN * 2,
            out_channels=NUM_DISC_KERN * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
            ),
        nn.BatchNorm2d(NUM_DISC_KERN * 4),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(
            in_channels=NUM_DISC_KERN * 4,
            out_channels=NUM_CHANNELS,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
            ),
        nn.Sigmoid()
        )

  def forward(self, input):
    return self.network(input)