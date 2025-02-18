from transforms import *

import os
import torchvision



def get_nbclasses(dataset):
    return {
        'cifar10': 10,
        'imagenet': 1000,
    }[dataset]

def get_dataset(dataset, split):
    return {
        'cifar10': get_cifar10,
        'imagenet': get_imagenet,
    }[dataset](split)



def get_cifar10(pflip, crop_scale,
                pjitter, pgray, brightness, contrast, saturation, hue,
                pblur, sigma):
    
    # pflip=0.5, crop_scale=(0.2, 1.0)
    # pjitter=0.8, pgray=0.2, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
    # pblur=0.5, sigma=(0.1, 2.0)

    inv_transform = ParamCompose([
        ParamRandomResizedCrop(pflip=pflip, size=(32, 32), scale=crop_scale, ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        ParamColorJitter(pjitter=pjitter, pgray=pgray, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        ParamGaussianBlur(pblur=pblur, kernel_size=tuple(2*round((x - 10)/20) + 1 for x in (224, 224)), sigma=sigma),
    ], [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    return AugDatasetWrapper(dataset, no_transform, inv_transform)



class AugDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, no_transform, inv_transform):
        self.dataset = dataset
        self.no_transform = no_transform
        self.inv_transform = inv_transform

        self.nb_params = inv_transform.nb_params

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx]

        img_0 = self.no_transform(img)
        img_1, param_1 = self.inv_transform(img)

        return (img_0, img_1, param_1)
    