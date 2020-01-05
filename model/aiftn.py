import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch import autograd
import time as t
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os
from src.tensorboard_logger import Logger
from itertools import chain
from torchvision import utils
import numpy as np

class Positive_Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = [80,80,3]
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),


            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=2, stride=2, padding=1))

        self.MSEloss = nn.MSELoss()




    def forward(self, x):
        x = self.main_module(x)
        return x


class Negative_Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        self.main_module = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),


            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=1024),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(num_features=512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=256),
            nn.LeakyReLU(0.2, inplace=True),

            # State (1024x4x4)
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(0.2, inplace=True),


            # State (256x16x16)
            nn.ConvTranspose2d(in_channels=64, out_channels=channels, kernel_size=2, stride=2, padding=1))

        self.MSEloss = nn.MSELoss()
        self.output = nn.Tanh()


    def forward(self, x):
        x = self.main_module(x)
        x = self.output(x)
        return x



class Image_Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=0))

    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class Frequency_Discriminator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        # Filters [256, 512, 1024]
        # Input_dim = channels (Cx64x64)
        # Output_dim = 1
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization --> using per instance normalization nn.InstanceNorm2d()
            # Image (Cx32x32)
            nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (256x16x16)
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(512, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # State (512x8x8)
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(1024, affine=True),
            nn.LeakyReLU(0.2, inplace=True))
        # output of main module --> State (1024x4x4)

        self.output = nn.Sequential(
            # The output of D is no longer a probability, we do not apply sigmoid at the output of D.
            nn.Conv2d(in_channels=1024, out_channels=1, kernel_size=3, stride=1, padding=0))


    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024*4*4)


class AIFTN(object):
    def __init__(self, args):
        print("Init AIFTN model.")
        self.GP = Positive_Generator(args.channels)
        self.GN = Negative_Generator(args.channels)
        self.DI = Image_Discriminator(args.channels)
        self.DF = Frequency_Discriminator(args.channels)
        self.C = args.channels
        self._lambda = args.recon_weight

        # Check if cuda is available
        self.check_cuda(cuda_flag=args.cuda,cuda_index=args.cuda_index)

        # WGAN values from paper
        self.learning_rate = 1e-4
        self.b1 = 0.5
        self.b2 = 0.999
        self.batch_size = args.batch_size

        # WGAN_gradient penalty uses ADAM
        self.di_optimizer = optim.Adam(self.DI.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.df_optimizer = optim.Adam(self.DF.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.gp_optimizer = optim.Adam(self.GP.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))
        self.gn_optimizer = optim.Adam(self.GN.parameters(), lr=self.learning_rate, betas=(self.b1, self.b2))

        # Set the logger
        self.logger = Logger('./logs')
        self.logger.writer.flush()
        self.number_of_images = 10

        self.generator_iters = args.generator_iters
        self.critic_iter = 5
        self.lambda_term = 10


    def check_cuda(self, cuda_flag=False,cuda_index = 0):
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.DI.cuda(self.cuda_index)
            self.GP.cuda(self.cuda_index)
            self.DF.cuda(self.cuda_index)
            self.GN.cuda(self.cuda_index)
            print("Cuda enabled flag: {}".format(self.cuda))

    def train(self, train_loader):
        self.t_begin = t.time()
        self.file = open("inception_score_graph.txt", "w")
        self._lambda = torch.tensor(self._lambda).cuda(self.cuda_index)
        # Now batches are callable self.data.next()
        #tmp  = self.get_infinite_batches(train_loader)
        self.img_data,self.freq_data,self.iter_count = self.get_infinite_batches(train_loader)
        self.img_data = iter(self.img_data)
        self.freq_data  = iter(self.freq_data)
        _iter = 0
        #self.img_data,self.freq_data = self.get_infinite_batches(train_loader)
        one = torch.FloatTensor([1])
        mone = one * -1
        if self.cuda:
            one = one.cuda(self.cuda_index)
            mone = mone.cuda(self.cuda_index)
        for g_iter in range(self.generator_iters):
            # Requires grad, Generator requires_grad = False
            for pi in self.DI.parameters():
                pi.requires_grad = True
            for pf in self.DF.parameters():
                pf.requires_grad = True
            dI_loss_real = 0
            dI_loss_fake = 0
            # Train Dicriminator forward-loss-backward-update self.critic_iter times while 1 Generator forward-loss-backward-update
            for d_iter in range(self.critic_iter):
                self.DI.zero_grad()
                self.DF.zero_grad()
                images = self.img_data.__next__()
                freqs = self.freq_data.__next__()
                _iter+=1
                if _iter==self.iter_count:
                    self.img_data, self.freq_data, self.iter_count = self.get_infinite_batches(train_loader)
                    self.img_data = iter(self.img_data)
                    self.freq_data = iter(self.freq_data)
                    _iter = 0
                freqs = freqs.type(torch.float)
                # Check for batch to have full batch_size
                if (images.size()[0] != self.batch_size):
                    continue
                if self.cuda:
                    images, freqs = Variable(images.cuda(self.cuda_index)), Variable(freqs.cuda(self.cuda_index))
                else:
                    images, freqs = Variable(images), Variable(freqs)

                # Train discriminators
                # Train with real images and Frequency

                di_loss_real = self.DI(images)
                di_loss_real = di_loss_real.mean()
                di_loss_real.backward(mone)

                df_loss_real = self.DF(freqs)
                df_loss_real = df_loss_real.mean()
                df_loss_real.backward(mone)


                #images = images.cuda(1)
                #freqs  = freqs.cuda(1)

                transformed_freqs = self.GP(images)
                transformed_images = self.GN(freqs)

                #transformed_freqs  = transformed_freqs.cuda(0)
                #transformed_images = transformed_images.cuda(0)


                di_loss_fake = self.DI(transformed_images)
                di_loss_fake = di_loss_fake.mean()
                di_loss_fake.backward(one)

                df_loss_fake = self.DF(transformed_freqs)
                df_loss_fake = df_loss_fake.mean()
                df_loss_fake.backward(one)

                img_gradient_penalty = self.calculate_gradient_penalty(images.data, transformed_images.data,img_or_freq='img')
                freq_gradient_penalty = self.calculate_gradient_penalty(freqs.data, transformed_freqs.data,img_or_freq='freq')

                img_gradient_penalty.backward(retain_graph=True)
                freq_gradient_penalty.backward(retain_graph=True)



                total_di_loss = di_loss_fake - di_loss_real + img_gradient_penalty
                total_df_loss = df_loss_fake - df_loss_real + freq_gradient_penalty
                total_d_loss = total_di_loss+ total_df_loss
                self.di_optimizer.step()
                self.df_optimizer.step()

            # Generator update
            for pi in self.DI.parameters():
                pi.requires_grad = False  # to avoid computation
            for pf in self.DF.parameters():
                pf.requires_grad = False  # to avoid computation

            self.GP.zero_grad()
            self.GN.zero_grad()
            images = images.cuda(self.cuda_index)
            freqs = freqs.cuda(self.cuda_index)


            #Generator

            #Computer adversarial loss
            transformed_freqs = self.GP(images)
            adv_gp_loss = self.DF(transformed_freqs)
            gp_loss = adv_gp_loss.mean()

            recon_gp_loss = self.GP.MSEloss(freqs,transformed_freqs)
            gp_loss += (self._lambda*recon_gp_loss)

            gp_loss.backward(mone)
            gp_cost = -gp_loss
            self.gp_optimizer.step()



            transformed_images = self.GN(freqs)
            adv_gn_loss = self.DI(transformed_images)
            gn_loss = adv_gn_loss.mean()

            recon_gn_loss = self.GN.MSEloss(images, transformed_images)
            gn_loss += (self._lambda*recon_gn_loss)


            gn_loss.backward(mone)
            gn_cost = -gn_loss
            self.gp_optimizer.step()
            # Saving model and sampling images every 1000th generator iterations
            if (g_iter) % 50 == 0:
                self.save_model()
                if not os.path.exists('training_result_vis/'):
                    os.makedirs('training_result_vis/')


                images = images.data.cpu()
                freqs = freqs.data.cpu()
                transformed_images = transformed_images.data.cpu()
                transformed_freqs = transformed_freqs.data.cpu()
                img_grid = utils.make_grid(images)
                utils.save_image(img_grid, 'training_result_vis/img_iter_{}.png'.format(str(g_iter).zfill(3)))

                freq_grid = utils.make_grid(200*freqs)
                utils.save_image(freq_grid,
                                 'training_result_vis/freqs_iter_{}.png'.format(str(g_iter).zfill(3)))

                timg_grid = utils.make_grid(transformed_images)
                utils.save_image(timg_grid,
                                 'training_result_vis/img_transformed_iter_{}.png'.format(str(g_iter).zfill(3)))

                tfreqs_grid = utils.make_grid(200*transformed_freqs)
                utils.save_image(tfreqs_grid,
                                 'training_result_vis/freq_transformed_iter_{}.png'.format(str(g_iter).zfill(3)))

                # Testing
                time = t.time() - self.t_begin
                #print("Real Inception score: {}".format(inception_score))
                print("Generator iter: {}".format(g_iter))
                print("Time {}".format(time))

                # Write to file inception_score, gen_iters, time
                #output = str(g_iter) + " " + str(time) + " " + str(inception_score[0]) + "\n"
                #self.file.write(output)
                # ============ TensorBoard logging ============#
                # (1) Log the scalar values
                info = {
                    'Loss DI': total_di_loss.data,
                    'Loss DF': total_df_loss.data,
                    'Loss D': total_d_loss.data,
                    'Loss GP': gp_cost.data,
                    'Loss GN': gn_cost.data,
                    'Loss GP Recon':recon_gp_loss.data,
                    'Loss GN Recon':recon_gn_loss.data,
                    'Loss DI Real': di_loss_real.data,
                    'Loss DI Fake': di_loss_fake.data,
                    'Loss DF Real': df_loss_real.data,
                    'Loss DF Fake': df_loss_fake.data

                }
                for tag, value in info.items():
                    self.logger.scalar_summary(tag, value, g_iter + 1)

                # (3) Log the images
                info = {
                    'real_images': self.real_images(images, self.number_of_images),
                    'real_frequencies': self.real_images(freqs, self.number_of_images),
                    'transformed_images': self.generate_sample(freqs, self.number_of_images,img_or_freq='img'),
                    'transformed_frequencies': self.generate_sample(images, self.number_of_images,img_or_freq='freq')
                }
                for tag, images in info.items():
                    self.logger.image_summary(tag, images, g_iter + 1)
        self.t_end = t.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        #self.file.close()
        # Save the trained parameters
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        z = Variable(torch.randn(self.batch_size, 100, 1, 1)).cuda(self.cuda_index)
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'dgan_model_image.png'.")
        utils.save_image(grid, 'dgan_model_image.png')

    def calculate_gradient_penalty(self, real_images, fake_images,img_or_freq = 'img'):
        if img_or_freq=='img':
            self.gp_D = self.DI
        if img_or_freq=='freq':
            self.gp_D = self.DF
        eta = torch.FloatTensor(self.batch_size,1,1,1).uniform_(0,1)
        eta = eta.expand(self.batch_size, real_images.size(1), real_images.size(2), real_images.size(3))
        if self.cuda:
            eta = eta.cuda(self.cuda_index)
        else:
            eta = eta

        interpolated = eta * real_images + ((1 - eta) * fake_images)

        if self.cuda:
            interpolated = interpolated.cuda(self.cuda_index)
        else:
            interpolated = interpolated

        # define it to calculate gradient
        interpolated = Variable(interpolated, requires_grad=True)

        # calculate probability of interpolated examples
        prob_interpolated = self.gp_D(interpolated)

        # calculate gradients of probabilities with respect to examples
        gradients = autograd.grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(
                                   prob_interpolated.size()).cuda(self.cuda_index) if self.cuda else torch.ones(
                                   prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.lambda_term
        return grad_penalty



    def real_images(self, images, number_of_images):
        if (self.C == 3):
            return self.to_np(images.view(-1, self.C, 80, 80)[:self.number_of_images])
        else:
            return self.to_np(images.view(-1, 80, 80)[:self.number_of_images])

    def generate_sample(self, input, number_of_images,img_or_freq = 'img'):
        input = input.cuda(self.cuda_index)
        if img_or_freq=='img':
            samples = self.GN(input).data.cpu().numpy()[:number_of_images]
        if img_or_freq=='freq':
            samples = self.GP(input).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            if self.C == 3:
                generated_images.append(sample.reshape(self.C,80, 80))
            else:
                generated_images.append(sample.reshape(80, 80))
        return generated_images



    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.GP.state_dict(), './positive_generator.pkl')
        torch.save(self.GN.state_dict(), './negative_generator.pkl')
        torch.save(self.DI.state_dict(), './image_discriminator.pkl')
        torch.save(self.DF.state_dict(), './frequency_discriminator.pkl')
        print('SAVE Model ')

    def load_model(self, D_model_filenames, G_model_filenames):
        """
        :param D_model_filenames: [0] is the discrminator for image,[1] is the discriminator for frequency
        :param G_model_filenames:
        :return:
        """
        DI_model_path = os.path.join(os.getcwd(), D_model_filenames[0])
        DF_model_path = os.path.join(os.getcwd(), D_model_filenames[1])

        GP_model_path = os.path.join(os.getcwd(), G_model_filenames[0])
        GN_model_path = os.path.join(os.getcwd(), G_model_filenames[0])

        self.DI.load_state_dict(torch.load(DI_model_path))
        self.DF.load_state_dict(torch.load(DF_model_path))
        self.GP.load_state_dict(torch.load(GP_model_path))
        self.GN.load_state_dict(torch.load(GN_model_path))

        print('Models are loaded.')

    def get_infinite_batches(self, data_loader):
        img_list = []
        freq_list = []
        iter_count = 0
        for img,freq in data_loader:
            img_list.append(img)
            freq_list.append(freq)
            iter_count +=1
        return img_list,freq_list,iter_count


