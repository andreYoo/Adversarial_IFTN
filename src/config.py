import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Pytorch implementation of 'Image-to-Frequency Transform Network for Road Defect Detection.")
    parser.add_argument('--is_train', type=str, default='True')
    parser.add_argument('--dataroot', type=str, default='/media/neumann/Warehouse/crack_DB/pavement crack datasets/CFD/cfd_image', help='path to dataset')
    parser.add_argument('--gtroot', type=str, default='/media/neumann/Warehouse/crack_DB/pavement crack datasets/CFD/seg_gt', help='path to dataset')
    parser.add_argument('--dataset', type=str, default='CFD', choices=['CFD'],help='The name of dataset')
    parser.add_argument('--epochs', type=int, default=50, help='The number of epochs to run')
    parser.add_argument('--channels', type=int, default=3, help='The number of channel of input')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--recon_weight', type=float, default=0.01, help='The size of batch')
    parser.add_argument('--cuda',  type=str, default='True', help='Availability of cuda')
    parser.add_argument('--cuda_index',  type=int, default=1, help='Index for cuda device')
    parser.add_argument('--load_D', type=str, default='False', help='Path for loading Discriminator network')
    parser.add_argument('--load_G', type=str, default='False', help='Path for loading Generator network')
    parser.add_argument('--generator_iters', type=int, default=10000, help='The number of iterations for generators in the model.')
    return check_args(parser.parse_args())

# Checking arguments
def check_args(args):
    # --epoch
    try:
        assert args.epochs >= 1
    except:
        print('Number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('Batch size must be larger than or equal to one')

    return args
