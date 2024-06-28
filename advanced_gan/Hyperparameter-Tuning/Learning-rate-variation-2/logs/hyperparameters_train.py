import os
import random
import wandb
import torch
import torchvision.utils as vutils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from hyperparameters_config import *
from hyperparameter_utils import *


def run():

    wandb.init(
        project='BASELINE-DCGAN',
        config={
        'num_samples': NUM_SAMPLES,
        'run_name': RUN_NAME,
        'seed': SEED,
        'batch_size': BATCH_SIZE,
        'gen_learning_rate': GEN_LEARNING_RATE,
        'disc_learning_rate': DISC_LEARNING_RATE,
        'beta1': BETA1,
        'beta2': BETA2,
        'latent_dims': LATENT_DIMS,
        'num_gen_kern': NUM_GEN_KERN,
        'num_disc_kern': NUM_DISC_KERN,
        'num_channels': NUM_CHANNELS,
        'epochs': EPOCHS
    })

    config = wandb.config

    if not os.path.exists('/logs'):
      os.mkdir('/logs')
    if not os.path.exists(f'/logs/{config.run_name}'):
      os.mkdir(f'/logs/{config.run_name}')

    manual_seed = config.seed
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    if torch.cuda.is_available():
      torch.cuda.manual_seed(manual_seed)
      torch.cuda.manual_seed_all(manual_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_loader = prepare_data(config.batch_size)

    generator, discriminator, gen_optimizer, disc_optimizer, calc_loss = model_setup()

    gen_losses = []
    disc_losses = []
    real_confs = []
    fake_confs = []
    generated_confs = []

    discriminator.train()
    generator.train()

    for epoch in range(1, config.epochs + 1):

        for batch_idx, (images, labels) in enumerate(tqdm(train_loader)):
            actual_images, real_batch_size, real_label, fake_label = train_setup(images, discriminator)

            # 1: train discrim
            reset_gradients(discriminator)

            out = discriminator(actual_images).view(-1)
            real_loss = calc_loss(out, real_label)
            real_loss.backward()
            real_conf = real_loss.mean().item()


            # 2: full pass on generated images
            noise = torch.randn(config.batch_size, config.latent_dims, 1, 1, device=DEVICE)

            fake_images = generator(noise)
            out = discriminator(fake_images.detach()).view(-1)
            fake_loss = calc_loss(out, fake_label)
            fake_loss.backward()
            fake_conf = fake_loss.mean().item()
            discriminator_loss = real_loss + fake_loss
            disc_optimizer.step()

            # 3: train generator
            reset_gradients(generator)

            out = discriminator(fake_images).view(-1)
            generator_loss = calc_loss(out, real_label)
            generator_loss.backward()
            generated_conf = generator_loss.mean().item()

            gen_optimizer.step()

            s_metrics = {
                'discriminator_loss': discriminator_loss.item(),
                'generator_loss': generator_loss.item(),
                'real_conf': real_conf,
                'fake_conf': fake_conf,
                'generated_conf': generated_conf
            }

            gen_losses.append(generator_loss.item())
            disc_losses.append(discriminator_loss.item())
            real_confs.append(real_conf)
            fake_confs.append(fake_conf)
            generated_confs.append(generated_conf)

        discriminator_loss = sum(disc_losses) / len(disc_losses)
        generator_loss = sum(gen_losses) / len(gen_losses)
        real_conf = sum(real_confs) / len(real_confs)
        fake_conf = sum(fake_confs) / len(fake_confs)
        generated_conf = sum(generated_confs) / len(generated_confs)

        e_metrics = {
            'discriminator_loss': discriminator_loss,
            'generator_loss': generator_loss,
            'real_conf': real_conf,
            'fake_conf': fake_conf,
            'generated_conf': generated_conf,
        }
        wandb.log({**s_metrics, **e_metrics})
        print(f'Epoch ', epoch, '/', config.epochs, ': Discriminator Loss: ', sum(disc_losses) / len(disc_losses), 'Generator: ', sum(gen_losses) / len(gen_losses))

        samples = []
        static_noise = torch.randn(real_batch_size, config.latent_dims, 1, 1, device=DEVICE)
        generator.eval()
        samples = generator(static_noise)

        grid = vutils.make_grid(samples[:24], normalize=True, scale_each=True)
        grid = grid.permute(1, 2, 0).cpu().numpy()
        grid = (grid * 255).astype(np.uint8)
        wandb.log({'samples': wandb.Image(grid)})

        if epoch % 5 == 0 or epoch == 1:
            plt.figure(figsize=(10, 10))
            for x in range(25):
                sample = samples[x]
                sample = np.reshape(sample.cpu().detach().numpy(), (28, 28))
                plt.subplot(5, 5, x + 1)
                plt.imshow(sample, cmap='gray')
                plt.axis('off')
            plt.show()
        generator.train()

        torch.save(generator.state_dict(), f'/logs/{config.run_name}/generator_{epoch}.pth')
        torch.save(discriminator.state_dict(), f'/logs/{config.run_name}/discriminator_{epoch}.pth')

        torch.cuda.empty_cache()
    wandb.finish()
    return generator, discriminator