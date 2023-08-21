from tsmnet.modules import Autoencoder

from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from pathlib import Path
import yaml
import torch
import os


def get_default_device():
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model(path, device=get_default_device()):
    """
    Args:
        mel2wav_path (str or Path): path to the root folder of dumped text2mel
        device (str or torch.device): device to load the model
    """
    root = Path(path)
    with open(os.path.join(os.path.dirname(path), "args.yml"), "r") as f:
        args = yaml.unsafe_load(f)
    netA = Autoencoder([int(n) for n in args.compress_ratios], args.ngf, args.n_residual_layers).to(device)
    netA.load_state_dict(torch.load(path, map_location=device))
    return netA


class Neuralgram:
    def __init__(
        self,
        path,
        device=None,
    ):
        if device is None:
            device = get_default_device()
        self.device = device
        self.netA = load_model(path, device)

    def __call__(self, audio):
        """
        Performs audio to neuralgram conversion (See Autoencoder.encoder in tsmnet/modules.py)
        Args:
            audio (torch.tensor): PyTorch tensor containing audio (batch_size, timesteps)
        Returns:
            torch.tensor: neuralgram computed on input audio (batch_size, channels, timesteps)
        """
        with torch.no_grad():
            return self.netA.encoder(torch.as_tensor(audio).unsqueeze(1).to(self.device))

    def inverse(self, neu):
        """
        Performs neuralgram to audio conversion
        Args:
            neu (torch.tensor): PyTorch tensor containing neuralgram (batch_size, channels, timesteps)
        Returns:
            torch.tensor:  Inverted raw audio (batch_size, timesteps)

        """
        with torch.no_grad():
            return self.netA.decoder(neu.to(self.device)).squeeze(1)

class Stretcher:
    def __init__(self, path, device=None):
        self.neuralgram = Neuralgram(path, device)
        
    @torch.no_grad()
    def __call__(self, audio, rate , interpolation=InterpolationMode.BICUBIC): # NEAREST | BILINEAR | BICUBIC
        if rate == 1:
            return audio.numpy() if isinstance(audio, torch.Tensor) else audio
        neu = self.neuralgram(audio)
        neu_resized = resize(
            neu,
            (*neu.shape[1:-1], int(neu.shape[-1] * (1/rate))),
            interpolation
        )
        return self.neuralgram.inverse(neu_resized)
