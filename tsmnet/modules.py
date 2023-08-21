import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm
import numpy as np


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


def WNConvTranspose1d(*args, **kwargs):
    return weight_norm(nn.ConvTranspose1d(*args, **kwargs))

class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Tanh(),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.Tanh(),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)
    
class Autoencoder(nn.Module):
    def __init__(self, compress_ratios, ngf, n_residual_layers):
        super().__init__()
        
        self.encoder = self.makeEncoder(compress_ratios, ngf, n_residual_layers)
        self.decoder = self.makeDecoder([r for r in reversed(compress_ratios)], ngf, n_residual_layers)
        
        self.apply(weights_init)
        
    def makeEncoder(self, ratios, ngf, n_residual_layers):
        mult = 1

        model = [
            nn.ReflectionPad1d(3),
            WNConv1d(1, ngf, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        
        # Downsample to neuralgram scale
        for i, r in enumerate(ratios):
            mult *= 2
            
            for j in range(n_residual_layers-1, -1, -1):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]
                
            model += [
                nn.Tanh(),
                WNConv1d(
                    mult * ngf // 2,
                    mult * ngf,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2
                ),
            ]

        model += [ nn.Tanh() ]
        
        return nn.Sequential(*model)
    def makeDecoder(self, ratios, ngf, n_residual_layers):
        mult = int(2 ** len(ratios))

        model = []

        # Upsample to raw audio scale
        for i, r in enumerate(ratios):
            model += [
                nn.Tanh(),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2
                ),
            ]

            for j in range(n_residual_layers):
                model += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

        model += [
            nn.Tanh(),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, 1, kernel_size=7, padding=0),
            nn.Tanh(),
        ]

        return nn.Sequential(*model)
    
    def forward(self, x):
        return self.decoder(self.encoder(x))

class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor):
        super().__init__()
        model = nn.ModuleDict()

        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(1, ndf, kernel_size=15),
            nn.Tanh(),
        )

        nf = ndf
        stride = downsampling_factor
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, 1024)

            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.Tanh(),
            )

        nf = min(nf * 2, 1024)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
        )

        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )

        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    def __init__(self, num_D, ndf, n_layers, downsampling_factor):
        super().__init__()
        self.model = nn.ModuleDict()
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor
            )

        self.downsample = nn.AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results
