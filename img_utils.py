import numpy as np
import torchvision

def get_reverse_transform(no_transform):
    mean = no_transform.transforms[-1].mean
    std = no_transform.transforms[-1].std

    reverse_mean = [- m for m in mean]
    reverse_std = [1 / s for s in std]

    return torchvision.transforms.Compose([
                        torchvision.transforms.Normalize((0, 0, 0), reverse_std),
                        torchvision.transforms.Normalize(reverse_mean, (1., 1., 1.)),
                        # torchvision.transforms.ToPILImage()
                    ])
