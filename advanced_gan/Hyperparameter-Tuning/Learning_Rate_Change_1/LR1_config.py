import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 12
RUN_NAME = 'lr_variation'
SEED = 0
BATCH_SIZE = 32
GEN_LEARNING_RATE = 0.005
DISC_LEARNING_RATE = 0.0001
BETA1 = 0.5
BETA2 = 0.999
LATENT_DIMS = 100
NUM_GEN_KERN = 128
NUM_DISC_KERN = 128
NUM_CHANNELS = 3
EPOCHS = 20