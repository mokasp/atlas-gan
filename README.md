#  Project
This repository covers the basics of Generative Adversarial Networks using Pytorch. 
It has two parts, but both . The first is a Deep Convolutional Generative Adveserial Network. 
# General Advesarial Networks
General Adverserial Networks are a class of ML frameworks that can be used to generate synthetic but realistic data from an existing dataset. They are made of two distinct neural networks:

| Network  | Goal | Input | Output |
| :------------ | ------------- | ------------- | ------------- |
| **Generator**  | create data that is indistinguishable from real data, fooling the discriminator into thinking the generated data is real  | random noise vector, usually sampled from gaussian or uniform distribution  | image of the same dimensions of the real images  |
| **Discriminator**  | correctly identify whether the input data is real or generated.  | either the real images from the dataset or the generated images from the generator  | probability value indicating whether or the input data is real (close to 1) or generated (close to 0) |



[dataset](https://www.kaggle.com/datasets/spandan2/cats-faces-64x64-for-generative-models)
