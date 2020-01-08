from src.config import parse_args
from model import aiftn
import torchvision.transforms as transforms
from src import utils,dataset
from torch.utils.data import Dataset, DataLoader
import h5py
import pdb
train_filename = 'CDF_train_Cropped_dataset.h5'
eval_filename = 'CDF_eval_Cropped_dataset.h5'


def main(args):
    #Model initilization
    model = aiftn.AIFTN(args)

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


    img_path = args.dataroot
    gt_path = args.gtroot


    # Start model training
    if args.is_train == 'True':

        # If there is no training dataset.
        # Load datasets to train and test loaders
        print('GENERATING TRAINING SAMPLES')
        #train_image, train_freq = utils.ext_trainset(img_path, gt_path)
        # eature_extraction = FeatureExtractionTest(train_loader, test_loader, args.cuda, args.batch_size)

        # If there is
        db = h5py.File(train_filename,'r')
        train_image = db['image']
        train_freq = db['frequency']
        train_loader = DataLoader(dataset.imgdataset(imgset=train_image, freqset=train_freq, transform=trans),
                                  batch_size=args.batch_size, shuffle=True, num_workers=4)
        print('MODEL TRAINING')
        model.train(train_loader)

    # start evaluating on test data

    else:
        print('MODEL EVAL')
        D_models_path = ['./image_discriminator.pkl','./frequency_discriminator.pkl']
        G_models_path = ['./positive_generator.pkl','./negative_generator.pkl']
        print('GENERATING EVAL SAMPLES')
        #eval_image, eval_freq = utils.ext_evalset(img_path, gt_path)


        db = h5py.File(eval_filename,'r')
        eval_image = db['image']
        eval_freq = db['frequency']

        eval_loader = DataLoader(dataset.imgdataset(imgset=eval_image, freqset=eval_freq, transform=trans),
                                 batch_size=args.batch_size, shuffle=True, num_workers=4)
        model.evaluate(eval_loader,D_model_path=D_models_path,G_model_path=G_models_path)



if __name__ == '__main__':
    args = parse_args()
    main(args)