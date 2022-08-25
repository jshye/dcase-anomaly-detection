"""
Class definition of AutoEncoder in PyTorch.

Copyright (C) 2021 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import torch
from torch import nn
import torch.nn.functional as F
import torchaudio


class Wav2vec_base(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        encoder_projection_dropout: float = 0.1
        encoder_attention_dropout: float = 0.1
        encoder_ff_interm_dropout: float = 0.1
        encoder_dropout: float = 0.1
        encoder_layer_drop: float = 0.1
        self.encoder = torchaudio.models.wav2vec2_model(
                            extractor_mode="group_norm",
                            extractor_conv_layer_config=None,
                            extractor_conv_bias=False,
                            encoder_embed_dim=768,
                            encoder_projection_dropout=encoder_projection_dropout,
                            encoder_pos_conv_kernel=128,
                            encoder_pos_conv_groups=16,
                            encoder_num_layers=12,
                            encoder_num_heads=12,
                            encoder_attention_dropout=encoder_attention_dropout,
                            encoder_ff_interm_features=3072,
                            encoder_ff_interm_dropout=encoder_ff_interm_dropout,
                            encoder_dropout=encoder_dropout,
                            encoder_layer_norm_first=False,
                            encoder_layer_drop=encoder_layer_drop,
                            aux_num_out=None,
                        )
        # [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

        # decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(768, 512, 2, 2, bias=False), 
            nn.ConvTranspose1d(512, 512, 2, 2, bias=False), 
            nn.ConvTranspose1d(512, 512, 3, 2, bias=False), 
            nn.ConvTranspose1d(512, 512, 3, 2, bias=False), 
            nn.ConvTranspose1d(512, 512, 3, 2, bias=False), 
            nn.ConvTranspose1d(512, 512, 3, 2, bias=False), 
            nn.ConvTranspose1d(512, 1, 10, 5, bias=False))

        self.criterion = nn.MSELoss()

    def forward(self, x):
        length = x.shape[-1]
        x = self.encoder(x)[0] # (batch, 499, 768)
        x = F.pad(x, (0, 0, 0, 1)).transpose(-2,-1)
        x = self.decoder(x)[...,:length].squeeze(-2)
        return x

    def get_loss(self, inputs):
        """
        Calculate loss function of AutoEncoder.
        """
        recon_x = self.forward(inputs)
        recon_loss = self.criterion(recon_x, inputs)

        return recon_loss