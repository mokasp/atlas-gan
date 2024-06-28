import torch

# force pytorch to use GPU if available
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

NUM_SAMPLES = 12
RUN_NAME = 'baseline'
SEED = 0

# values for model dimensions
LATENT_DIMS = 100
NUM_GEN_KERN = 128
NUM_DISC_KERN = 128
NUM_CHANNELS = 1
BATCH_SIZE = 128

# parameters for training
LEARNING_RATE = 0.0001
BETA1 = 0.5
BETA2 = 0.999
EPOCHS = 20