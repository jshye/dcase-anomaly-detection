"""
https://github.com/openvinotoolkit/anomalib/tree/development/anomalib/models/ganomaly
"""

import math
from typing import Tuple, Union
import matplotlib.pyplot as plt

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def pad_nextpow2(batch: Tensor) -> Tensor:
    """Compute required padding from input size and return padded images.
    Finds the largest dimension and computes a square image of dimensions that are of the power of 2.
    In case the image dimension is odd, it returns the image with an extra padding on one side.
    Args:
        batch (Tensor): Input images
    Returns:
        batch: Padded batch
    """
    # find the largest dimension
    l_dim = 2 ** math.ceil(math.log(max(*batch.shape[-2:]), 2))
    padding_w = [math.ceil((l_dim - batch.shape[-2]) / 2), math.floor((l_dim - batch.shape[-2]) / 2)]
    padding_h = [math.ceil((l_dim - batch.shape[-1]) / 2), math.floor((l_dim - batch.shape[-1]) / 2)]
    padded_batch = F.pad(batch, pad=[*padding_h, *padding_w])
    return padded_batch


def plot_recons(orig, real, fake, anomaly=False, epoch=None, show=False):
    fig = plt.figure(figsize=(12,4))

    # vmin = min(np.min(real.cpu().numpy()[0,0,:,:]), np.min(fake.cpu().detach().numpy()[0,0,:,:]))
    # vmax = max(np.max(real.cpu().numpy()[0,0,:,:]), np.max(fake.cpu().detach().numpy()[0,0,:,:]))

    plt.subplot(1,3,1)
    plt.imshow(orig.cpu()[0,0,:,:])
    plt.colorbar(shrink=0.5)
    plt.title('source')

    plt.subplot(1,3,2)
    # plt.imshow(real.cpu()[0,0,:,:], vmin=vmin, vmax=vmax)
    plt.imshow(real.cpu()[0,0,:,:])
    plt.colorbar(shrink=0.5)
    plt.title('real (generator input)')

    plt.subplot(1,3,3)
    # plt.imshow(fake.cpu()[0,0,:,:])
    # plt.imshow(fake.cpu().detach().numpy()[0,0,:,:], vmin=vmin, vmax=vmax)
    plt.imshow(fake.cpu().detach().numpy()[0,0,:,:])
    plt.colorbar(shrink=0.5)
    plt.title('fake')

    title = ''
    if anomaly:
        title += 'Anomaly'
    else:
        title += 'Normal'

    if epoch:
        title += f' - epoch {epoch}'

    plt.suptitle(title)
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        return fig


def plot_latent(latent_i, latent_o, anomaly=False, epoch=None, show=False):
    fig = plt.figure(figsize=(12,28))
    # vmin_i = np.min(latent_i.cpu().detach().numpy())
    # vmax_i = np.max(latent_i.cpu().detach().numpy())
    # vmin_o = np.min(latent_o.cpu().detach().numpy())
    # vmax_o = np.max(latent_o.cpu().detach().numpy())

    anomaly_score = torch.mean(torch.pow((latent_i - latent_o), 2), dim=[1,2,3])

    for i in range(100):
        pos = 2*i + 1
        plt.subplot(20,10,pos)
        plt.imshow(latent_i.cpu().detach().numpy()[0,i,:,:])
        plt.title(f'i - {i}')
        plt.axis('off')

    for i in range(100):
        pos = 2*i + 2
        plt.subplot(20,10,pos)
        plt.imshow(latent_o.cpu().detach().numpy()[0,i,:,:])
        plt.title(f'o - {i}')
        plt.axis('off')

    # plt.colorbar(shrink=0.5)
    # plt.suptitle(f'Normal-latent_i (vmin: {vmin_i:.4f}, vmax: {vmax_i:.4f})')
    title = ''
    if anomaly:
        title += 'Anomaly'
    else:
        title += 'Normal'

    title += f' {anomaly_score.item():.4f}'

    if epoch:
        title += f' - epoch {epoch}'

    plt.suptitle(title, fontsize='x-large', y=1.02)
    plt.tight_layout()
    
    if show:
        plt.show()
    else:
        return fig
        
################ Torch models defining encoder, decoder, Generator and Discriminator. #############

class Encoder(nn.Module):
    """Encoder Network.
    Args:
        input_size (Tuple[int, int]): Size of input image
        latent_vec_size (int): Size of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ):
        super().__init__()

        self.input_layers = nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channels}-{n_features}",
            nn.Conv2d(num_input_channels, n_features, kernel_size=4, stride=2, padding=4, bias=False),
        )
        self.input_layers.add_module(f"initial-relu-{n_features}", nn.LeakyReLU(0.2, inplace=True))

        # Extra Layers
        self.extra_layers = nn.Sequential()

        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_features}-conv",
                nn.Conv2d(n_features, n_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-batchnorm", nn.BatchNorm2d(n_features))
            self.extra_layers.add_module(f"extra-layers-{layer}-{n_features}-relu", nn.LeakyReLU(0.2, inplace=True))

        # Create pyramid features to reach latent vector
        self.pyramid_features = nn.Sequential()
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_features
            out_features = n_features * 2
            self.pyramid_features.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv2d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False),
            )
            self.pyramid_features.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.pyramid_features.add_module(f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            n_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Final conv
        if add_final_conv_layer:
            self.final_conv_layer = nn.Conv2d(
                n_features,
                latent_vec_size,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            )

    def forward(self, input_tensor: Tensor):
        """Return latent vectors."""

        output = self.input_layers(input_tensor)
        output = self.extra_layers(output)
        output = self.pyramid_features(output)
        if self.final_conv_layer is not None:
            output = self.final_conv_layer(output)

        return output


class Decoder(nn.Module):
    """Decoder Network.
    Args:
        input_size (Tuple[int, int]): Size of input image
        latent_vec_size (int): Size of latent vector z
        num_input_channels (int): Number of input channels in the image
        n_features (int): Number of features per convolution layer
        extra_layers (int): Number of extra layers since the network uses only a single encoder layer by default.
            Defaults to 0.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
    ):
        super().__init__()

        self.latent_input = nn.Sequential()

        # Calculate input channel size to recreate inverse pyramid
        exp_factor = math.ceil(math.log(min(input_size) // 2, 2)) - 2
        n_input_features = n_features * (2**exp_factor)

        # CNN layer for latent vector input
        self.latent_input.add_module(
            f"initial-{latent_vec_size}-{n_input_features}-convt",
            nn.ConvTranspose2d(
                latent_vec_size,
                n_input_features,
                kernel_size=4,
                stride=1,
                padding=0,
                bias=False,
            ),
        )
        self.latent_input.add_module(f"initial-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features))
        self.latent_input.add_module(f"initial-{n_input_features}-relu", nn.ReLU(True))

        # Create inverse pyramid
        self.inverse_pyramid = nn.Sequential()
        pyramid_dim = min(*input_size) // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 4:
            in_features = n_input_features
            out_features = n_input_features // 2
            self.inverse_pyramid.add_module(
                f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose2d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm2d(out_features))
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-relu", nn.ReLU(True))
            n_input_features = out_features
            pyramid_dim = pyramid_dim // 2

        # Extra Layers
        self.extra_layers = nn.Sequential()
        for layer in range(extra_layers):
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-conv",
                nn.Conv2d(n_input_features, n_input_features, kernel_size=3, stride=1, padding=1, bias=False),
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-batchnorm", nn.BatchNorm2d(n_input_features)
            )
            self.extra_layers.add_module(
                f"extra-layers-{layer}-{n_input_features}-relu", nn.LeakyReLU(0.2, inplace=True)
            )

        # Final layers
        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f"final-{n_input_features}-{num_input_channels}-convt",
            nn.ConvTranspose2d(
                n_input_features,
                num_input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.final_layers.add_module(f"final-{num_input_channels}-tanh", nn.Tanh())

    def forward(self, input_tensor):
        """Return generated image."""
        output = self.latent_input(input_tensor)
        output = self.inverse_pyramid(output)
        output = self.extra_layers(output)
        output = self.final_layers(output)
        return output


class Discriminator(nn.Module):
    """Discriminator.
        Made of only one encoder layer which takes x and x_hat to produce a score.
    Args:
        input_size (Tuple[int,int]): Input image size.
        num_input_channels (int): Number of image channels.
        n_features (int): Number of feature maps in each convolution layer.
        extra_layers (int, optional): Add extra intermediate layers. Defaults to 0.
    """

    def __init__(self, input_size: Tuple[int, int], num_input_channels: int, n_features: int, extra_layers: int = 0):
        super().__init__()
        encoder = Encoder(input_size, 1, num_input_channels, n_features, extra_layers)
        layers = []
        for block in encoder.children():
            if isinstance(block, nn.Sequential):
                layers.extend(list(block.children()))
            else:
                layers.append(block)

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, input_tensor):
        """Return class of object and features."""
        features = self.features(input_tensor)
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)
        return classifier, features


class Generator(nn.Module):
    """Generator model.
    Made of an encoder-decoder-encoder architecture.
    Args:
        input_size (Tuple[int,int]): Size of input data.
        latent_vec_size (int): Dimension of latent vector produced between the first encoder-decoder.
        num_input_channels (int): Number of channels in input image.
        n_features (int): Number of feature maps in each convolution layer.
        extra_layers (int, optional): Extra intermediate layers in the encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add a final convolution layer in the decoder. Defaults to True.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        latent_vec_size: int,
        num_input_channels: int,
        n_features: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ):
        super().__init__()
        self.encoder1 = Encoder(
            input_size, latent_vec_size, num_input_channels, n_features, extra_layers, add_final_conv_layer
        )
        self.decoder = Decoder(input_size, latent_vec_size, num_input_channels, n_features, extra_layers)
        self.encoder2 = Encoder(
            input_size, latent_vec_size, num_input_channels, n_features, extra_layers, add_final_conv_layer
        )

    def forward(self, input_tensor):
        """Return generated image and the latent vectors."""
        latent_i = self.encoder1(input_tensor)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return gen_image, latent_i, latent_o


class GanomalyModel(nn.Module):
    """Ganomaly Model.
    Args:
        input_size (Tuple[int,int]): Input dimension.
        num_input_channels (int): Number of input channels.
        n_features (int): Number of features layers in the CNNs.
        latent_vec_size (int): Size of autoencoder latent vector.
        extra_layers (int, optional): Number of extra layers for encoder/decoder. Defaults to 0.
        add_final_conv_layer (bool, optional): Add convolution layer at the end. Defaults to True.
    """

    def __init__(
        self,
        input_size: Tuple[int, int],
        num_input_channels: int,
        n_features: int,
        latent_vec_size: int,
        extra_layers: int = 0,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()
        self.generator: Generator = Generator(
            input_size=input_size,
            latent_vec_size=latent_vec_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
            add_final_conv_layer=add_final_conv_layer,
        )
        self.discriminator: Discriminator = Discriminator(
            input_size=input_size,
            num_input_channels=num_input_channels,
            n_features=n_features,
            extra_layers=extra_layers,
        )
        self.weights_init(self.generator)
        self.weights_init(self.discriminator)

        self.tanh = nn.Tanh()  ## normalize input [-1, 1]  --> different from original source

    @staticmethod
    def weights_init(module: nn.Module):
        """Initialize DCGAN weights.
        Args:
            module (nn.Module): [description]
        """
        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def forward(self, batch: Tensor) -> Union[Tuple[Tensor, Tensor, Tensor, Tensor], Tensor]:
        """Get scores for batch.
        Args:
            batch (Tensor): Images
        Returns:
            Tensor: Regeneration scores.
        """
        batch = self.tanh(batch)  ## normalize input [-1, 1]  --> different from original source
        padded_batch = pad_nextpow2(batch)
        fake, latent_i, latent_o = self.generator(padded_batch)
        # if self.training:
        #     return padded_batch, fake, latent_i, latent_o
        # return torch.mean(torch.pow((latent_i - latent_o), 2), dim=1).view(-1)  # convert nx1x1 to n
        return padded_batch, fake, latent_i, latent_o

##################################### Loss Functions #####################################

class GeneratorLoss(nn.Module):
    """Generator loss for the GANomaly model.
    Args:
        wadv (int, optional): Weight for adversarial loss. Defaults to 1.
        wcon (int, optional): Image regeneration weight. Defaults to 50.
        wenc (int, optional): Latent vector encoder weight. Defaults to 1.
    """

    def __init__(self, wadv=1, wcon=50, wenc=1):
        super().__init__()

        self.loss_enc = nn.SmoothL1Loss()
        self.loss_adv = nn.MSELoss()
        self.loss_con = nn.L1Loss()

        self.wadv = wadv
        self.wcon = wcon
        self.wenc = wenc

    def forward(
        self, latent_i: Tensor, latent_o: Tensor, images: Tensor, fake: Tensor, pred_real: Tensor, pred_fake: Tensor
    ) -> Tensor:
        """Compute the loss for a batch.
        Args:
            latent_i (Tensor): Latent features of the first encoder.
            latent_o (Tensor): Latent features of the second encoder.
            images (Tensor): Real image that served as input of the generator.
            fake (Tensor): Generated image.
            pred_real (Tensor): Discriminator predictions for the real image.
            pred_fake (Tensor): Discriminator predictions for the fake image.
        Returns:
            Tensor: The computed generator loss.
        """
        error_enc = self.loss_enc(latent_i, latent_o)
        error_con = self.loss_con(images, fake)
        error_adv = self.loss_adv(pred_real, pred_fake)

        loss = error_adv * self.wadv + error_con * self.wcon + error_enc * self.wenc
        return loss


class DiscriminatorLoss(nn.Module):
    """Discriminator loss for the GANomaly model."""

    def __init__(self):
        super().__init__()

        self.loss_bce = nn.BCELoss()

    def forward(self, pred_real, pred_fake):
        """Compye the loss for a predicted batch.
        Args:
            pred_real (Tensor): Discriminator predictions for the real image.
            pred_fake (Tensor): Discriminator predictions for the fake image.
        Returns:
            Tensor: The computed discriminator loss.
        """
        error_discriminator_real = self.loss_bce(
            pred_real, torch.ones(size=pred_real.shape, dtype=torch.float32, device=pred_real.device)
        )
        error_discriminator_fake = self.loss_bce(
            pred_fake, torch.zeros(size=pred_fake.shape, dtype=torch.float32, device=pred_fake.device)
        )
        loss_discriminator = (error_discriminator_fake + error_discriminator_real) * 0.5
    
        # return
        return loss_discriminator