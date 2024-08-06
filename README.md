# Autoencoder Architectures with PyTorch

This notebook is a personal attempt to code various autoencoder architectures using PyTorch. It includes implementations of linear, convolutional linear, non-linear convolutional, and multi-layer perceptron (MLP) autoencoders on the MNIST dataset. The goal of this project is to improve my coding skills in PyTorch and deepen my understanding of essential deep learning concepts.

## Features

- **Linear Autoencoder**: Simple autoencoder with fully connected layers.
- **Convolutional Linear Autoencoder**: Combines convolutional and linear layers.
- **Non-linear Convolutional Autoencoder**: Utilizes non-linear activation functions in the convolutional architecture.
- **MLP Autoencoder**: Employs a multi-layer perceptron structure.

## Insights

- **Compression Tool**: Autoencoders are indeed very powerful as a tool for data compression, reducing the dimensionality of input data while retaining essential information.
- **Generative Model Limitations**: The notebook demonstrates that, without a structured latent space, these autoencoders cannot function effectively as generative models. This limitation sets the stage for future work with Variational Autoencoders (VAEs) to explore their generative capabilities.

## Code Structure

- **Versatile Code**: The code is designed to be flexible and adaptable, providing an object-oriented structure for autoencoders suitable for larger projects.
- **Autoencoder Object**: A single, versatile `AutoEncoder` class is implemented to handle different encoder and decoder architectures, making it easy to extend and integrate into other projects.

### `AutoEncoder` Class

```python
import torch.nn as nn

class AutoEncoder(nn.Module):

    def __init__(self, encoder, decoder, device='cpu'):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)

        self.optimizer = None
        self.criterion = None
        self.device = device
        self.to(device)

    def forward(self, x):
        x = x.to(self.device)
        z = self.encoder(x)
        x_prime = self.decoder(z)
        return z, x_prime
    
    def encode(self, x):
        x = x.to(self.device)
        return self.encoder(x)
    
    def decode(self, z):
        z = z.to(self.device)
        return self.decoder(z)
    
    def train_step(self, training_batch):
        self.train()
        self.optimizer.zero_grad()
        
        x = training_batch.to(self.device)
        z, x_prime = self.forward(x)
        
        loss = self.criterion(x_prime, x)
        loss.backward()
        self.optimizer.step()
        
        return loss
```

## Next Steps

Variational Autoencoders (VAEs): The next step is to implement VAEs to create a structured latent space, allowing the model to generate new data samples using stochastic encoders and decoders.

By exploring these different autoencoder architectures, I aim to build a strong foundation in PyTorch and gain practical insights into deep learning model design and implementation.
