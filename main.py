from src.config import parse_args
from model import aiftn
import torchvision.transforms as transforms
from src import utils,dataset
from torch.utils.data import Dataset, DataLoader
import h5py
import pdb
filename = 'CDF_Cropped_dataset.h5'


def main(args):
    #Model initilization
    model = aiftn.AIFTN(args)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load datasets to train and test loaders
    print('GENERATING TRAINING SAMPLES')
    img_path = args.dataroot
    gt_path = args.gtroot

    #Generate image path
    train_image,train_freq = utils.ext_trainset(img_path,gt_path)
    #eature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

    # db = h5py.File(filename,'r')
    # train_image = db['image']
    # train_freq = db['frequency']

    train_loader = DataLoader(dataset.imgdataset(imgset=train_image,freqset=train_freq,transform=trans),batch_size=args.batch_size,shuffle=True,num_workers=4)

    # Start model training
    if args.is_train == 'True':
        print('MODEL TRAINING')
        model.train(train_loader)

    # start evaluating on test data

    else:
        print('MODEL EVAL')
        #model.evaluate(test_loader)



if __name__ == '__main__':
    args = parse_args()
    main(args)