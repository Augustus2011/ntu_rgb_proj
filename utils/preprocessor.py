import PIL
import numpy as np
import torch
import torchvision
from torchvision import transforms

def preprocess(img_path:str):
    a_image=PIL.Image.open(img_path)
    a_image=a_image.convert("RGB")
    preprocess = transforms.Compose([
    transforms.ToTensor(),
])
    return preprocess(a_image)
