import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from base_networks import *
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import utils
from logger import Logger


class Net(torch.nn.Module):
    def __init__(self, num_channels, base_filter):
        super(Net, self).__init__()

        self.layers = torch.nn.Sequential(
            ConvBlock(num_channels, base_filter, 9, 1, 0, norm=None),
            ConvBlock(base_filter, base_filter // 2, 1, 1, 0, norm=None),
            ConvBlock(base_filter // 2, num_channels, 5, 1, 0, activation=None, norm=None),
        )

    def forward(self, x):
        out = self.layers(x)
        return out

    def weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)


class SRCNN(object):
    def __init__(self, args):
        # parameters
        self.model_name = args.model_name
        self.dataset = args.dataset
        self.num_threads = args.num_threads
        self.num_channels = args.num_channels
        self.scale_factor = args.scale_factor
        self.num_epochs = args.num_epochs
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode

        if self.dataset == 'bsds300':
            self.image_size = 256
        elif self.dataset == 'mnist':
            self.image_size = 28
        elif self.dataset == 'celebA':
            self.image_size = 64

    def load_dataset(self, dataset='train'):
        if dataset == 'train':
            print('Loading train datasets...')
            train_set = get_training_set(self.data_dir, self.dataset, self.image_size, self.scale_factor, is_gray=False,
                                         normalize=False)
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)
        elif dataset == 'test':
            print('Loading test datasets...')
            test_set = get_test_set(self.data_dir, self.dataset, self.image_size, self.scale_factor, is_gray=False,
                                    normalize=False)
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False)

    def train(self):
        # networks
        self.model = Net(num_channels=self.num_channels, base_filter=64)

        # weigh initialization
        self.model.weight_init(mean=0.0, std=0.001)

        # optimizer
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

        # loss function
        if self.gpu_mode:
            self.model.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.model)
        print('----------------------------------------------')

        # load dataset
        train_data_loader = self.load_dataset(dataset='train')
        test_data_loader = self.load_dataset(dataset='test')

        # set the logger
        log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = Logger(log_dir)

        ################# Train #################
        print('Training is started.')
        avg_loss = []
        step = 0

        # test image
        test_input, test_target = test_data_loader.__iter__().__next__()

        self.model.train()
        for epoch in range(self.num_epochs):

            epoch_loss = 0
            for iter, (input, target) in enumerate(train_data_loader):
                # input data (bicubic interpolated image)
                if self.gpu_mode:
                    x_ = Variable(target.cuda())
                    y_ = Variable(utils.img_interp(input).cuda())
                else:
                    x_ = Variable(target)
                    y_ = Variable(utils.img_interp(input))

                # update network
                self.optimizer.zero_grad()
                recon_image = self.model(y_)

                # exclude border pixels from loss computation
                padding = 6
                x_crop = utils.to_var(utils.to_np(x_)[:, :, padding:-padding, padding:-padding])
                loss = self.MSE_loss(recon_image, x_crop)
                loss.backward()
                self.optimizer.step()

                # log
                epoch_loss += loss.data[0]
                print("Epoch: [%2d] [%4d/%4d] loss: %.8f" % ((epoch + 1), (iter + 1), len(train_data_loader), loss.data[0]))

                # tensorboard logging
                logger.scalar_summary('loss', loss.data[0], step + 1)
                step += 1

            # avg. loss per epoch
            avg_loss.append(epoch_loss / len(train_data_loader))

            # prediction
            recon_img = self.model(Variable(utils.img_interp(test_input).cuda()))
            recon_img = recon_img[0].cpu().data
            gt_img = test_target[0]
            lr_img = test_input[0]
            bc_img = utils.img_interp(lr_img)

            # calculate psnrs
            bc_psnr = utils.PSNR(bc_img, gt_img)
            recon_psnr = utils.PSNR(recon_img, gt_img)

            # save result images
            result_imgs = [gt_img, lr_img, bc_img, recon_img]
            psnrs = [None, bc_psnr, recon_psnr]
            utils.plot_test_result(result_imgs, psnrs, epoch, save_dir=self.save_dir)

            print("Saving training result images at epoch %d" % epoch)

        # Plot avg. loss
        utils.plot_loss([avg_loss], self.num_epochs, save_dir=self.save_dir)
        print("Training is finished.")

        # Save trained parameters of model
        self.save_model()

    def test(self):
        # networks
        self.model = Net(num_channels=self.num_channels, base_filter=64)

        if self.gpu_mode:
            self.model.cuda()

        # load model
        self.load_model()

        # load dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Test
        print('Test is started.')
        img_num = 0
        self.model.eval()
        for input, target in test_data_loader:
            # input data (bicubic interpolated image)
            if self.gpu_mode:
                y_ = Variable(utils.img_interp(input).cuda())
            else:
                y_ = Variable(utils.img_interp(input))

            # prediction
            recon_imgs = self.model(y_)
            padding = 6
            for i, recon_img in enumerate(recon_imgs):
                img_num += 1
                recon_img = recon_img.cpu().data[:, padding:-padding, padding:-padding]
                gt_img = target[i][:, padding:-padding, padding:-padding]
                lr_img = input[i][:, padding:-padding, padding:-padding]
                bc_img = utils.img_interp(lr_img)


                # calculate psnrs
                bc_psnr = utils.PSNR(bc_img, gt_img)
                recon_psnr = utils.PSNR(recon_img, gt_img)

                # save result images
                result_imgs = [gt_img, lr_img, bc_img, recon_img]
                psnrs = [None, bc_psnr, recon_psnr]
                utils.plot_test_result(result_imgs, psnrs, img_num, save_dir=self.save_dir)

                print("Saving %d test result images..." % img_num)

    def save_model(self):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        torch.save(self.model.state_dict(), model_dir + '/' + self.model_name + '_param.pkl')
        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.save_dir, 'model')

        model_name = model_dir + '/' + self.model_name + '_param.pkl'
        if os.path.exists(model_name):
            self.model.load_state_dict(torch.load(model_name))
            print('Trained generator model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False
