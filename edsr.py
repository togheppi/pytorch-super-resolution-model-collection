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
    def __init__(self, num_channels, base_filter, num_residuals):
        super(Net, self).__init__()

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation=None, norm=None)

        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResnetBlock(base_filter, norm=None))
        self.residual_layers = nn.Sequential(*resnet_blocks)

        self.mid_conv = ConvBlock(base_filter, base_filter, 3, 1, 1, activation=None, norm=None)

        self.upscale4x = nn.Sequential(
            Upsample2xBlock(base_filter, base_filter, upsample='ps', activation=None, norm=None),
            Upsample2xBlock(base_filter, base_filter, upsample='ps', activation=None, norm=None),
        )

        self.output_conv = ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)

    def forward(self, x):
        out = self.input_conv(x)
        residual = out
        out = self.residual_layers(out)
        out = self.mid_conv(out)
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.output_conv(out)
        return out


class EDSR(object):
    def __init__(self, args):
        # parameters
        self.model_name = args.model_name
        self.train_dataset = args.train_dataset
        self.test_dataset = args.test_dataset
        self.crop_size = args.crop_size
        self.num_threads = args.num_threads
        self.num_channels = args.num_channels
        self.scale_factor = args.scale_factor
        self.num_epochs = args.num_epochs
        self.save_epochs = args.save_epochs
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr = args.lr
        self.data_dir = args.data_dir
        self.save_dir = args.save_dir
        self.gpu_mode = args.gpu_mode

    def load_dataset(self, dataset='train'):
        if self.num_channels == 1:
            is_gray = True
        else:
            is_gray = False

        if dataset == 'train':
            print('Loading train datasets...')
            train_set = get_training_set(self.data_dir, self.train_dataset, self.crop_size, self.scale_factor, is_gray=is_gray,
                                         normalize=False)
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)
        elif dataset == 'test':
            print('Loading test datasets...')
            test_set = get_test_set(self.data_dir, self.test_dataset, self.scale_factor, is_gray=is_gray,
                                    normalize=False)
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False)

    def train(self):
        # networks
        self.model = Net(num_channels=self.num_channels, base_filter=64, num_residuals=16)

        # weigh initialization
        self.model.weight_init()

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999), eps=1e-8)

        # loss function
        if self.gpu_mode:
            self.model.cuda()
            self.L1_loss = nn.L1Loss().cuda()
        else:
            self.L1_loss = nn.L1Loss()

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
        test_input, test_target = test_data_loader.dataset.__getitem__(2)
        test_input = test_input.unsqueeze(0)
        test_target = test_target.unsqueeze(0)

        self.model.train()
        for epoch in range(self.num_epochs):

            # learning rate is decayed by a factor of 2 every 40 epochs
            if (epoch+1) % 40 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] /= 2.0
                print("Learning rate decay: lr={}".format(self.optimizer.param_groups[0]["lr"]))

            epoch_loss = 0
            for iter, (input, target) in enumerate(train_data_loader):
                # input data (low resolution image)
                if self.gpu_mode:
                    x_ = Variable(target.cuda())
                    y_ = Variable(input.cuda())
                else:
                    x_ = Variable(target)
                    y_ = Variable(input)

                # update network
                self.optimizer.zero_grad()
                recon_image = self.model(y_)
                loss = self.L1_loss(recon_image, x_)
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
            recon_imgs = self.model(Variable(test_input.cuda()))
            recon_img = recon_imgs[0].cpu().data
            gt_img = test_target[0]
            lr_img = test_input[0]
            bc_img = utils.img_interp(test_input[0], self.scale_factor)

            # calculate psnrs
            bc_psnr = utils.PSNR(bc_img, gt_img)
            recon_psnr = utils.PSNR(recon_img, gt_img)

            # save result images
            result_imgs = [gt_img, lr_img, bc_img, recon_img]
            psnrs = [None, None, bc_psnr, recon_psnr]
            utils.plot_test_result(result_imgs, psnrs, epoch + 1, save_dir=self.save_dir, is_training=True)

            print("Saving training result images at epoch %d" % (epoch + 1))

            # Save trained parameters of model
            if (epoch + 1) % self.save_epochs == 0:
                self.save_model(epoch + 1)

        # Plot avg. loss
        utils.plot_loss([avg_loss], self.num_epochs, save_dir=self.save_dir)
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def test(self):
        # networks
        self.model = Net(num_channels=self.num_channels, base_filter=64, num_residuals=16)

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
            # input data (low resolution image)
            if self.gpu_mode:
                y_ = Variable(input.cuda())
            else:
                y_ = Variable(input)

            # prediction
            recon_imgs = self.model(y_)
            for i, recon_img in enumerate(recon_imgs):
                img_num += 1
                recon_img = recon_imgs[i].cpu().data
                gt_img = target[i]
                lr_img = input[i]
                bc_img = utils.img_interp(input[i], self.scale_factor)

                # calculate psnrs
                bc_psnr = utils.PSNR(bc_img, gt_img)
                recon_psnr = utils.PSNR(recon_img, gt_img)

                # save result images
                result_imgs = [gt_img, lr_img, bc_img, recon_img]
                psnrs = [None, None, bc_psnr, recon_psnr]
                utils.plot_test_result(result_imgs, psnrs, img_num, save_dir=self.save_dir)

                print("Saving %d test result images..." % img_num)

    def save_model(self, epoch=None):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if epoch is not None:
            torch.save(self.model.state_dict(), model_dir + '/' + self.model_name + '_param_epoch_%d.pkl' % epoch)
        else:
            torch.save(self.model.state_dict(), model_dir + '/' + self.model_name + '_param.pkl')

        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.save_dir, 'model')

        model_name = model_dir + '/' + self.model_name + '_param.pkl'
        if os.path.exists(model_name):
            self.model.load_state_dict(torch.load(model_name))
            print('Trained model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False
