#  Project
This repository covers the basics of Generative Adversarial Networks using Pytorch. It has two parts, a DCGAN and a GAN but both follow the same structure. I started with a baseline model and from that model I tested different variations of the network by adjusting the architecture or using different parameters to see if any improvements could be made. 

# General Advesarial Networks
General Adverserial Networks are a class of ML frameworks that can be used to generate synthetic but realistic data from an existing dataset. They are made of two distinct neural networks:

| Network  | Goal | Input | Output |
| :------------ | ------------- | ------------- | ------------- |
| **Generator**  | create data that is indistinguishable from real data, fooling the discriminator into thinking the generated data is real  | random noise vector, usually sampled from gaussian or uniform distribution  | image of the same dimensions of the real images  |
| **Discriminator**  | correctly identify whether the input data is real or generated.  | either the real images from the dataset or the generated images from the generator  | probability value indicating whether or the input data is real (close to 1) or generated (close to 0) |

During each step of training, the discriminator is trained and updated first with the real images from the dataset, and then the generator is used to make the fake images to train and update the discriminator with the fake images. Afterwards, the generator is updated. This process continues until the discriminator can no longer correctly distinguish the fake images from the real ones.

## DCGAN
![dcgan_samples png](https://github.com/user-attachments/assets/a3ca8320-4456-46d8-9ff5-c432c39f5a47)
The Deep Convolutional Generative Advesarial Network is very similar but uses convolutional layers instead of dense layers. This model is trained on the MNIST dataset, and its goal is to generate realistic "handwritten" digits. 

## GAN
![gan_samples](https://github.com/user-attachments/assets/479a99e2-18f7-4a19-9865-c0d0693c94eb)


[dataset](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models)
