"""
Class definition of MobileNetV2 in PyTorch.

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
from torchvision import models


class MobileNetV2(nn.Module):
    """
    MobileNetV2.

    official repo.
    https://github.com/pytorch/vision/blob/master/torchvision/models/mobilenetv2.py
    """

    def __init__(self, num_sections):
        super().__init__()

        self.model = models.mobilenet_v2(
            pretrained=False,  # random initialization
            num_classes=num_sections,  # 1000 (default) -> 3 (00, 01, and 02)
            width_mult=0.5,  # expand ratio
        )

        # dropout is removed and output is unnormalized logits.
        self.model.classifier = nn.Linear(self.model.last_channel, num_sections)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs):
        "Forward propagation through MobileNetV2."

        # duplication; 1ch -> 3ch
        # shape: (512, 1, 64, 128) into (512, 3, 64, 128)
        dup = torch.cat((inputs, inputs, inputs), dim=1)
        output = self.model(dup)

        return output

    def get_loss(self, inputs, labels):
        "Calculate loss function through MobileNetV2."

        output = self.forward(inputs)
        import pdb; pdb.set_trace()
        loss = self.criterion(output, labels)

        return loss
