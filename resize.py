from tqdm import tqdm
import os
import argparse
from PIL import Image


def get_img_id(list_path):
    img_ids = [i_id.strip() for i_id in open(list_path)]
    return img_ids

def resize_data(img_path, mask_path, target_path, name):
    try:
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

        #Scale
        height, width = image.size
        max_size = 1024
        if height > width:
            r = max_size / height
            dim = (max_size, int(width * r))
        else:
            r = max_size / width
            dim = (int(height * r), max_size)
        image = image.resize(dim, Image.BICUBIC)
        mask = mask.resize(dim, Image.BICUBIC)
        target = target.resize(dim, Image.BICUBIC)

        image.save(img_file)
        mask.save(mask_file)
        target.save(target_file)
    except:
        print("An exception occurred")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='',
                        help='directory for dataset images')

    parser.add_argument('--mask_path', type=str, default='',
                        help='directory for saving images')
    
    parser.add_argument('--target_path', type=str, default='',
                        help='directory for saving images')
    parser.add_argument('--list_path', type=str, default='',
                        help='directory for saving images')
    args = parser.parse_args()

    img_ids = get_img_id(args.list_path)

    for img_id in tqdm(img_ids):
        resize_data(args.img_path, args.mask_path, args.target_path, img_id)
