from transforms import *

import os
import torchvision



def get_nbclasses(dataset):
    return {
        'cifar10': 10,
        'imagenet': 1000,
    }[dataset]

def get_dataset(device, dataset, split):
    # return {
    #     'cifar10': get_cifar10,
    #     'imagenet': get_imagenet,
    # }[dataset](device, split)
    return get_cifar10(device, split)



def get_cifar10(device, split,
                # pflip, crop_scale,
                # pjitter, pgray, brightness, contrast, saturation, hue,
                # pblur, sigma
                pflip=0.5, crop_scale=(0.2, 1.0),
                pjitter=0.8, pgray=0.2, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1,
                pblur=0.5, sigma=(0.1, 2.0)
               ):

    dataset = torchvision.datasets.CIFAR10(root='../EquiMod/data', train=(split == 'train'), download=True)

    no_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
        ])

    transform = ParamCompose([
        ParamRandomResizedCrop(pflip=pflip, size=(32, 32), scale=crop_scale, ratio=(3./4., 4./3.), interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
        ParamColorJitter(pjitter=pjitter, pgray=pgray, brightness=brightness, contrast=contrast, saturation=saturation, hue=hue),
        # ParamGaussianBlur(pblur=pblur, kernel_size=tuple(2*round((x - 10)/20) + 1 for x in (224, 224)), sigma=sigma),
    ], [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ])

    return AugDatasetWrapper(dataset, no_transform, transform, device)



class AugDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset, no_transform, transform, device):
        self.dataset = dataset
        self.no_transform = no_transform
        self.transform = transform

        self.device = device

        self.source = "cifar10"
        self.p_mean, self.p_std = self.get_normalizers()

        self.nb_params = transform.nb_params
    
    def get_normalizers(self):
        if self.source == 'cifar10':
            p_mean = torch.tensor([[4.3122e+00, 4.3216e+00, 2.3369e+01, 2.3374e+01, 4.9998e-01, 8.0087e-01,
                                    1.2025e+00, 1.4007e+00, 1.5964e+00, 1.8004e+00, 9.9993e-01, 1.0002e+00,
                                    9.9986e-01, 2.2321e-07, 1.9965e-01]])
            p_std = torch.tensor([[3.9740, 3.9851, 4.9544, 4.9539, 0.5000, 0.3993, 1.1651, 1.0210, 1.0200,
                                   1.1669, 0.2066, 0.2068, 0.2066, 0.0517, 0.3997]])
        elif self.source == 'imagenet':
            p_mean = torch.tensor([[6.8162e+01, 9.9199e+01, 2.6933e+02, 2.7457e+02, 4.9905e-01, 8.0054e-01,
                                    1.1998e+00, 1.3994e+00, 1.6014e+00, 1.7995e+00, 1.0001e+00, 1.0000e+00,
                                    1.0005e+00, 1.5640e-04, 2.0018e-01, 5.0030e-01, 5.2507e-01]])
            p_std = torch.tensor([[7.7370e+01, 9.8681e+01, 1.3686e+02, 1.4349e+02, 5.0000e-01, 3.9959e-01,
                                   1.1661e+00, 1.0201e+00, 1.0201e+00, 1.1657e+00, 4.1347e-01, 4.1323e-01,
                                   4.1349e-01, 1.0333e-01, 4.0013e-01, 5.0000e-01, 6.5251e-01]])
        return p_mean, p_std

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, _ = self.dataset[idx] # 2nd return variable (label) not used here

        img_0 = self.no_transform(img)
        img_t, params = self.transform(img)
        params = (params - self.p_mean) / self.p_std
        params = torch.squeeze(params)

        return img_0, img_t, params

    def update_tau(self, tau):
        self.transform.update_tau(tau)