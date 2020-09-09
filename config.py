import argparse

def test_parsers():
    parser = argparse.ArgumentParser("test code for image harmonization")
    # Basic options
    parser.add_argument('--img_path', type=str, default='Path_to_images',
                        help='Directory path to a batch of images')
    parser.add_argument('--mask_path', type=str, default='Path_to_masks',
                        help='Directory path to a batch of mask')
    parser.add_argument('--test_list_path', type=str, default='Path_to_test_list',
                        help='Directory path to test list')
    parser.add_argument('--target_path', type=str, default='Path_to_target',
                        help='Directory to a batch of target')
    parser.add_argument('--gpu_ids', type=str, default='-1',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model', type=str, default='./checkpoint/network.pth')
    args = parser.parse_args()
    return args

def train_parsers():
    parser = argparse.ArgumentParser("Pytorch training code for image harmonization")
    # Basic options
    parser.add_argument('--img_path', type=str, default='Path_to_images',
                        help='Directory path to images')
    parser.add_argument('--mask_path', type=str, default='Path_to_masks',
                        help='Directory path to mask')
    parser.add_argument('--list_path', type=str, default='Path_to_txt',
                        help='path to list')
    parser.add_argument('--target_path', type=str, default='Path_to_target',
                        help='Directory to target')
    # training options
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
    parser.add_argument('--model_save_dir', default='./checkpoint',
                        help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--epoch_size', type=int, default=230,
                        help="training epoch size")
    parser.add_argument('--n_workers', type=int, default=20)
    parser.add_argument('--seed', type=int, default=1,
                        help="random seed for training")
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training or not')
    parser.add_argument('--log_dir', default='./checkpoint/tensorboard',
                        help='Directory to save the tensorboardX')
    args = parser.parse_args()
    return args