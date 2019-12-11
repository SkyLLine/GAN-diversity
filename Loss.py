import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

def compute_loss(phi_fake):
    # print(phi_fake.shape)
    phi_fake = F.normalize(phi_fake,p=2,dim=0)
    phi_fake=torch.reshape(phi_fake,(phi_fake.size(0),1))
    phi = torch.mm(phi_fake,phi_fake.t())
    eye = torch.eye(phi_fake.size(0), device=phi_fake.device).byte()
    eye = torch.bitwise_not(eye).float()
    return (phi * eye).sum()