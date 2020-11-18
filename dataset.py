import os
import torch.utils.data as data
import random
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as transforms


class HarmDataSet(data.Dataset):
    def __init__(self, img_path, list_path, mask_path, target_path):
        super(HarmDataSet,self).__init__()
        self.img_path = img_path
        self.list_path = list_path
        self.mask_path = mask_path
        self.target_path = target_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []

        for name in self.img_ids:
            try:
                name_prepare = name.split("_", 2)
                mask_name = name_prepare[0] + '_' + name_prepare[1]
                img_name = name
                target_name = name_prepare[0]
                mask_file = os.path.join(self.mask_path, "%s.png" % mask_name)
                img_file = os.path.join(self.img_path, "%s" % img_name)
                target_file = os.path.join(self.target_path, "%s.jpg" % target_name)

                self.files.append({
                    "img": img_file,
                    "mask": mask_file,
                    "target": target_file,
                    "name": name
                })
            except:
                print("file doesn't exist")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        mask = Image.open(datafiles["mask"]).convert('1')
        target = Image.open(datafiles["target"]).convert('RGB')
        
        height, width = image.size
        min_size = random.randint(420, 512)
        if height > width:
            r = min_size / width
            dim = (int(height * r), min_size)
        else:
            r = min_size / height
            dim = (min_size, int(width * r))

        image = image.resize(dim, Image.BICUBIC)
        mask = mask.resize(dim, Image.BICUBIC)
        target = target.resize(dim, Image.BICUBIC)
        
        img_h, img_w = image.size
        nh = random.randint(0, img_h - 256)
        nw = random.randint(0, img_w - 256)
        image = image.crop((nh, nw, (nh + 256), (nw + 256)))
        mask = mask.crop((nh, nw, (nh + 256), (nw + 256)))
        target = target.crop((nh, nw, (nh + 256), (nw + 256)))
        
        if np.random.choice([True, False]):
            image = ImageOps.mirror(image)
            mask = ImageOps.mirror(mask)
            target = ImageOps.mirror(target)

        image = transforms.ToTensor()(image)
        mask = transforms.ToTensor()(mask)
        target = transforms.ToTensor()(target)

        return image, mask, target