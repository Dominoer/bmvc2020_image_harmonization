import torch
import numpy as np
from collections import OrderedDict
import os


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(model, epoch, filename, optimizer=None, save_arch=False, params=None):
    attributes = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }

    filename = os.path.join(filename, 'network.pth')

    if optimizer is not None:
        attributes['optimizer'] = optimizer.state_dict()

    if save_arch:
        attributes['arch'] = model

    if params is not None:
        attributes['params'] = params

    try:
        torch.save(attributes, filename)
    except TypeError:
        if 'arch' in attributes:
            print('Model architecture will be ignored because the architecture includes non-pickable objects.')
            del attributes['arch']
            torch.save(attributes, filename)

def load_checkpoint(path, model=None, optimizer=None):
    resume = torch.load(path)

    if model is not None:
        if ('module' in list(resume['state_dict'].keys())[0]) \
                and not (isinstance(model, torch.nn.DataParallel)):
            new_state_dict = OrderedDict()
            for k, v in resume['state_dict'].items():
                new_state_dict[k.replace('module.', '')] = v  # remove DataParallel wrapping

            model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(resume['state_dict'])

    if optimizer is not None:
        optimizer.load_state_dict(resume['optimizer'])

    return model, optimizer

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0))) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)