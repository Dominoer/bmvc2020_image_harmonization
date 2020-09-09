import os
import torch
import torchvision
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
import random
import utils
import model
import numpy as np
from config import train_parsers
from dataset import HarmDataSet

args = train_parsers()

random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#initialize path
model_save_dir = os.path.join(args.model_save_dir)
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

network = model.Network().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adadelta(network.parameters())

# restart the training process
if args.resume is True:
    network, optimizer = utils.load_checkpoint(model_save_path, network, optimizer=optimizer)

dst = HarmDataSet(args.img_path, args.list_path, args.mask_path, args.target_path)

trainloader = data.DataLoader(dst, batch_size=args.batch_size, shuffle=True, num_workers=args.n_workers, pin_memory=True)

for epoch in tqdm(range(args.epoch_size)):
    # to track the training loss as the model trains
    train_losses = utils.AverageMeter()
    network.train()
    for batch_index, (image, mask, target) in enumerate(trainloader):
        image, mask, target = image.to(device), mask.to(device), target.to(device)
        output = network(image, mask)
        loss = criterion(target, output)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.update(loss.item(), output.size(0))
    train_loss = train_losses.avg
    
    print('Training Loss: {}'.format(train_loss))

    utils.save_checkpoint(network, epoch, model_save_dir, optimizer)