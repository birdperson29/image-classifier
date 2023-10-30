import argparse
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils import data
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import json


def load_checkpoint(filepath):
    model = torchvision.models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    checkpoint = torch.load(filepath)
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']

    return model
def process_image(image_path):
    pil_image = Image.open(image_path)

    pil_image = pil_image.resize((256, 256))

    pil_image = pil_image.crop((16, 16, 240, 240))

    np_image = np.array(pil_image) / 255

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    np_image = np_image.transpose((2, 0, 1))

    tensor = torch.from_numpy(np_image)
    tensor = tensor.type(torch.FloatTensor)

    return tensor

def predict(image_path, model, topk, args.category_names_json_filepath):
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    image = process_image(image_path)
    image = image.unsqueeze(0)

    image = image.to(device)
    model.eval()
    with torch.no_grad():
        ps = torch.exp(model(image))

    ps, top_classes = ps.topk(topk, dim=1)

    idx_to_flower = {v: cat_to_name[k] for k, v in model.class_to_idx.items()}
    predicted_flowers_list = [idx_to_flower[i] for i in top_classes.tolist()[0]]

    return ps.tolist()[0], predicted_flowers_list

def print_predictions(args):
    model = load_checkpoint(args.model_filepath)
    model.to(device)

    with open(args.category_names_json_filepath, 'r') as f:
        cat_to_name = json.load(f, strict=False)

    top_ps, top_classes = predict(args.image_filepath, model, args.top_k, args)

    print("Predictions:")
    for i in range(args.top_k):
        print("#{: <3} {: <25} Prob: {:.2f}%".format(i, top_classes[i], top_ps[i] * 100)
              
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='image_filepath', help="image to be classified")
    parser.add_argument(dest='model_filepath', help="checkpoint file path")
    parser.add_argument('--category_names_json_filepath', dest='category_names_json_filepath', default='cat_to_name.json')
    parser.add_argument('--top_k', dest='top_k', help="most likely classes", default=5, type=int)
    parser.add_argument('--gpu', dest="gpu", action='store_true')

    args = parser.parse_args()
    print_predictions(args)
              
              
    