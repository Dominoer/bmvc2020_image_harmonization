from pathlib import Path
import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
import model
from config import *
import utils
from skimage import data, img_as_float
from skimage.measure import compare_mse as mse
from skimage.measure import compare_psnr as psnr


args = test_parsers()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_test_id(list_path):
    img_ids = [i_id.strip() for i_id in open(list_path)]
    return img_ids

model = model.Network().to(device)
model, _ = utils.load_checkpoint(args.model, model)
model.eval()

def get_test_data(img_path, mask_path, target_path, name):
    name_prepare = name.split("_", 2)
    mask_name = name_prepare[0] + '_' + name_prepare[1]
    img_name = name
    target_name = name_prepare[0]
    mask_file = os.path.join(mask_path, "%s.png" % mask_name)
    img_file = os.path.join(img_path, "%s" % img_name)
    target_file = os.path.join(target_path, "%s.jpg" % target_name)    
    image = Image.open(img_file).convert('RGB')
    mask = Image.open(mask_file).convert('1')
    target = Image.open(target_file).convert('RGB')
    dim = [256, 256]
    image = image.resize(dim, Image.BICUBIC)
    mask = mask.resize(dim, Image.BICUBIC)
    target = target.resize(dim, Image.BICUBIC)
    image = transforms.ToTensor()(image)
    mask = transforms.ToTensor()(mask)
    target = transforms.ToTensor()(target)
    
    return image, mask, target

img_ids = get_test_id(args.test_list_path)

with open("metrics.txt", "w") as f:
    for img_id in tqdm(img_ids):
        image, mask, target = get_test_data(args.img_path, args.mask_path, args.target_path, img_id)
        with torch.no_grad():
            image = image.to(device).unsqueeze(0)
            mask = mask.to(device).unsqueeze(0)
            target = target.to(device).unsqueeze(0)
            output = model(image, mask)
            output = utils.tensor2im(output, imtype=np.float32)
            target = utils.tensor2im(target, imtype=np.float32)
            mse_score_op = mse(output,target)
            psnr_score_op = psnr(target, output,data_range=output.max() - output.min())
            f.write('ID:{}, MSE:{}, PSNR:{}\n'.format(img_id, mse_score_op, psnr_score_op))
f.close()