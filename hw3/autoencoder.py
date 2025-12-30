import torch
import torch.nn as nn
import torch.nn.functional as F
from math import prod

class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement a CNN. Save the layers in the modules list.
        #  The input shape is an image batch: (N, in_channels, H_in, W_in).
        #  The output shape should be (N, out_channels, H_out, W_out).
        #  You can assume H_in, W_in >= 64.
        #  Architecture is up to you, but it's recommended to use at
        #  least 3 conv layers. You can use any Conv layer parameters,
        #  use pooling or only strides, use any activation functions,
        #  use BN or Dropout, etc.
        # ====== YOUR CODE: ======
        ndf = 128 #the initial output channels for the first convolutional layer. 

        # --- Layer 1 ---
        # Input: (N, in_channels, H, W) -> Output: (N, 128, H/2, W/2)
        modules.append(nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1, bias=False))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        # --- Layer 2 ---
        # Input: (N, 128, H/2, W/2) -> Output: (N, 256, H/4, W/4)
        modules.append(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(ndf * 2))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        # --- Layer 3 ---
        # Input: (N, 256, H/4, W/4) -> Output: (N, 512, H/8, W/8)
        modules.append(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(ndf * 4))
        modules.append(nn.LeakyReLU(0.2, inplace=True))

        # --- Layer 4 (Output Mapping) ---
        # Input: (N, 512, H/8, W/8) -> Output: (N, out_channels, H/16, W/16)
        modules.append(nn.Conv2d(ndf * 4, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(out_channels))
        modules.append(nn.LeakyReLU(0.2, inplace=True))
        
        #pass
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        modules = []

        # TODO:
        #  Implement the "mirror" CNN of the encoder.
        #  For example, instead of Conv layers use transposed convolutions,
        #  instead of pooling do unpooling (if relevant) and so on.
        #  The architecture does not have to exactly mirror the encoder
        #  (although you can), however the important thing is that the
        #  output should be a batch of images, with same dimensions as the
        #  inputs to the Encoder were.
        # ====== YOUR CODE: ======
        ngf = 128 #same as in the encoder

        # --- Layer 1 ---
        # Input: (N, in_channels, 4, 4) -> Output: (N, 512, 8, 8)
        modules.append(nn.ConvTranspose2d(in_channels, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(ngf * 4))
        modules.append(nn.ReLU(True)) 

        # --- Layer 2 ---
        # Input: (N, 512, 8, 8) -> Output: (N, 256, 16, 16)
        modules.append(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(ngf * 2))
        modules.append(nn.ReLU(True))

        # --- Layer 3 ---
        # Input: (N, 256, 16, 16) -> Output: (N, 128, 32, 32)
        modules.append(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False))
        modules.append(nn.BatchNorm2d(ngf))
        modules.append(nn.ReLU(True))

        # --- Layer 4 (Output) ---
        # Input: (N, 128, 32, 32) -> Output: (N, out_channels, 64, 64)
        modules.append(nn.ConvTranspose2d(ngf, out_channels, kernel_size=4, stride=2, padding=1, bias=False))
        
        #pass
        # ========================
        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder that extracts features from an input.
        :param features_decoder: Instance of a decoder that reconstructs an input from its features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)

        # TODO: Add more layers as needed for encode() and decode().
        # ====== YOUR CODE: ======
        self.fc_mu = nn.Linear(n_features, z_dim)
        self.fc_log_sigma2 = nn.Linear(n_features, z_dim)

        self.fc_z_project = nn.Linear(z_dim, n_features)
        #pass
        # ========================

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        # TODO:
        #  Sample a latent vector z given an input x from the posterior q(Z|x).
        #  1. Use the features extracted from the input to obtain mu and
        #     log_sigma2 (mean and log variance) of q(Z|x).
        #  2. Apply the reparametrization trick to obtain z.
        # ====== YOUR CODE: ======
        h = self.features_encoder(x)
        
        # 1. Flatten the output for the linear layers
        # (N, C, H, W) -> (N, C*H*W)
        h = h.view(h.size(0), -1)
        
        # 2. Calculate mu and log_sigma2 using the linear layers
        mu = self.fc_mu(h)       
        log_sigma2 = self.fc_log_sigma2(h) 
        
        # 3. Reparametrization Trick
        # We need standard deviation (std) for the formula: z = mu + std * epsilon
        # We have log(sigma^2). Remember: sigma = sqrt(sigma^2) = (sigma^2)^0.5
        # So: std = exp(log_sigma2 * 0.5)
        std = torch.exp(0.5 * log_sigma2)
        
        # 4. Sample epsilon from standard normal distribution
        # It must have the same shape and device as std
        eps = torch.randn_like(std)
        
        # 5. Calculate z
        z = mu + std * eps
        
        #pass
        # ========================
        return z, mu, log_sigma2

    def decode(self, z):
        # TODO:
        #  Convert a latent vector back into a reconstructed input.
        #  1. Convert latent z to features h with a linear layer.
        #  2. Apply features decoder.
        # ====== YOUR CODE: ======
        h = self.fc_z_project(z)
        h_cube = h.view(-1, *self.features_shape)
        x_rec = self.features_decoder(h_cube)
        # ========================
        # Scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            # TODO:
            #  Sample from the model. Generate n latent space samples and
            #  return their reconstructions.
            #  Notes:
            #  - Remember that this means using the model for INFERENCE.
            #  - We'll ignore the sigma2 parameter here:
            #    Instead of sampling from N(psi(z), sigma2 I), we'll just take
            #    the mean, i.e. psi(z).
            # ====== YOUR CODE: ======
            z = torch.randn(n, self.z_dim, device=device)
            x_generated = self.decode(z)
            
            for i in range(n):
                samples.append(x_generated[i])
            #pass
            # ========================
        # Detach and move to CPU for display purposes.
        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    loss, data_loss, kldiv_loss = None, None, None
    # TODO:
    #  Implement the VAE pointwise loss calculation.
    #  Remember:
    #  1. The covariance matrix of the posterior is diagonal.
    #  2. You need to average over the batch dimension.
    # ====== YOUR CODE: ======
    # --- 1. Data Loss (Reconstruction Loss) ---
    d_x = x[0].numel() 

    mse_per_sample = torch.sum((x - xr).pow(2).view(x.size(0), -1), dim=1)
    
    data_loss = torch.mean(mse_per_sample) / (x_sigma2 * d_x)

    # --- 2. KL Divergence Loss ---
    kld_per_sample = torch.sum(torch.exp(z_log_sigma2) + z_mu.pow(2) - 1 - z_log_sigma2, dim=1)
    kldiv_loss = torch.mean(kld_per_sample)

    # --- 3. Total Loss ---
    loss = data_loss + kldiv_loss
    #pass
    # ========================

    return loss, data_loss, kldiv_loss
