#!/usr/bin/env python
# coding: utf-8

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--pt7_path",   help="Path of *.pt7", type=str, required=True)
parser.add_argument("--batch_size", help="Batch size",    type=int, default=1)
parser.add_argument("--cifar",      help="Path of CIFAR", type=str, default='.')
args = parser.parse_args()

import torch
import net

torch.backends.cudnn.enabled = False
param = torch.load( args.pt7_path, map_location='cpu' )
wrn = net.Net( batch_size=args.batch_size, weights=param['params'], stats=param['stats'] )
# wrn.load_weight(param['params'])
# wrn.load_stats(param['stats'])
for param in wrn.parameters():
    param.requires_grad = False
print(wrn)
wrn.eval()

def normalize_unsqueeze( img, mean, std ):
    img_tensor = to_tensor(img).transpose(0,2)
    img_tensor = img_tensor-(torch.Tensor(mean)/256)
    img_tensor = img_tensor/(torch.Tensor( std)/256)
    img_tensor = img_tensor.transpose(0,2)
    return img_tensor.unsqueeze(0)


correct = 0
total = 0
import torchvision
cifar = torchvision.datasets.CIFAR10( args.cifar, train=False, download=True)
to_tensor = torchvision.transforms.ToTensor()
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(cifar):

        outputs = wrn( normalize_unsqueeze(inputs, [125.3, 123.0, 113.9], [63, 62.1, 66.7]) )
        _, predicted = outputs.max(1)
        total += 1
        correct += 1 if predicted==targets else 0#predicted.eq(targets).sum().item()
        print(correct,'/',total)
