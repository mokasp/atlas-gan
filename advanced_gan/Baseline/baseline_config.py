import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_SAMPLES = 12
RUN_NAME = 'baseline_gan'
SEED = 0
BATCH_SIZE = 32
LEARNING_RATE = 0.0005
BETA1 = 0.5
BETA2 = 0.999
LATENT_DIMS = 100
NUM_GEN_KERN = 128
NUM_DISC_KERN = 128
NUM_CHANNELS = 3
EPOCHS = 20
EPOCHS = 20