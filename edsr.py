import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from base_networks import *
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import utils
from logger import Logger
from torchvision.transforms import *


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

    def load_dataset(self, dataset, is_train=True):
        if self.num_channels == 1:
            is_gray = True
        else:
            is_gray = False

        if is_train:
            print('Loading train datasets...')
            train_set = get_training_set(self.data_dir, dataset, self.crop_size, self.scale_factor, is_gray=is_gray)
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)
        else:
            print('Loading test datasets...')
            test_set = get_test_set(self.data_dir, dataset, self.scale_factor, is_gray=is_gray)
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
        train_data_loader = self.load_dataset(dataset=self.train_dataset, is_train=True)
        test_data_loader = self.load_dataset(dataset=self.test_dataset[0], is_train=False)

        # set the logger
        log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        logger = Logger(log_dir)

        ################# Train #################
        print('Training is started.')
        avg_loss = []
        step = 0

        # test image
        test_lr, test_hr, test_bc = test_data_loader.dataset.__getitem__(2)
        test_lr = test_lr.unsqueeze(0)
        test_hr = test_hr.unsqueeze(0)
        test_bc = test_bc.unsqueeze(0)

        self.model.train()
        for epoch in range(self.num_epochs):

            # learning rate is decayed by a factor of 2 every 40 epochs
            if (epoch+1) % 40 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] /= 2.0
                print('Learning rate decay: lr={}'.format(self.optimizer.param_groups[0]['lr']))

            epoch_loss = 0
            for iter, (lr, hr, _) in enumerate(train_data_loader):
                # input data (low resolution image)
                if self.num_channels == 1:
                    x_ = Variable(hr[:, 0].unsqueeze(1))
                    y_ = Variable(lr[:, 0].unsqueeze(1))
                else:
                    x_ = Variable(hr)
                    y_ = Variable(lr)

                if self.gpu_mode:
                    x_ = x_.cuda()
                    y_ = y_.cuda()

                # update network
                self.optimizer.zero_grad()
                recon_image = self.model(y_)
                loss = self.L1_loss(recon_image, x_)
                loss.backward()
                self.optimizer.step()

                # log
                epoch_loss += loss.data[0]
                print('Epoch: [%2d] [%4d/%4d] loss: %.8f' % ((epoch + 1), (iter + 1), len(train_data_loader), loss.data[0]))

                # tensorboard logging
                logger.scalar_summary('loss', loss.data[0], step + 1)
                step += 1

            # avg. loss per epoch
            avg_loss.append(epoch_loss / len(train_data_loader))

            # prediction
            if self.num_channels == 1:
                y_ = Variable(test_lr[:, 0].unsqueeze(1))
            else:
                y_ = Variable(test_lr)

            if self.gpu_mode:
                y_ = y_.cuda()

            recon_img = self.model(y_)
            sr_img = recon_img[0].cpu().data

            # save result image
            save_dir = os.path.join(self.save_dir, 'train_result')
            utils.save_img(sr_img, epoch + 1, save_dir=save_dir, is_training=True)
            print('Result image at epoch %d is saved.' % (epoch + 1))

            # Save trained parameters of model
            if (epoch + 1) % self.save_epochs == 0:
                self.save_model(epoch + 1)

        # calculate psnrs
        if self.num_channels == 1:
            gt_img = test_hr[0][0].unsqueeze(0)
            lr_img = test_lr[0][0].unsqueeze(0)
            bc_img = test_bc[0][0].unsqueeze(0)
        else:
            gt_img = test_hr[0]
            lr_img = test_lr[0]
            bc_img = test_bc[0]

        bc_psnr = utils.PSNR(bc_img, gt_img)
        recon_psnr = utils.PSNR(sr_img, gt_img)

        # plot result images
        result_imgs = [gt_img, lr_img, bc_img, sr_img]
        psnrs = [None, None, bc_psnr, recon_psnr]
        utils.plot_test_result(result_imgs, psnrs, self.num_epochs, save_dir=save_dir, is_training=True)
        print('Training result image is saved.')

        # Plot avg. loss
        utils.plot_loss([avg_loss], self.num_epochs, save_dir=save_dir)
        print('Training is finished.')

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
        for test_dataset in self.test_dataset:
            test_data_loader = self.load_dataset(dataset=test_dataset, is_train=False)

            # Test
            print('Test is started.')
            img_num = 0
            total_img_num = len(test_data_loader)
            self.model.eval()
            for lr, hr, bc in test_data_loader:
                # input data (low resolution image)
                if self.num_channels == 1:
                    y_ = Variable(lr[:, 0].unsqueeze(1))
                else:
                    y_ = Variable(lr)

                if self.gpu_mode:
                    y_ = y_.cuda()

                # prediction
                recon_imgs = self.model(y_)
                for i, recon_img in enumerate(recon_imgs):
                    img_num += 1
                    sr_img = recon_img.cpu().data

                    # save result image
                    save_dir = os.path.join(self.save_dir, 'test_result', test_dataset)
                    utils.save_img(sr_img, img_num, save_dir=save_dir)

                    # calculate psnrs
                    if self.num_channels == 1:
                        gt_img = hr[i][0].unsqueeze(0)
                        lr_img = lr[i][0].unsqueeze(0)
                        bc_img = bc[i][0].unsqueeze(0)
                    else:
                        gt_img = hr[i]
                        lr_img = lr[i]
                        bc_img = bc[i]

                    bc_psnr = utils.PSNR(bc_img, gt_img)
                    recon_psnr = utils.PSNR(sr_img, gt_img)

                    # plot result images
                    result_imgs = [gt_img, lr_img, bc_img, sr_img]
                    psnrs = [None, None, bc_psnr, recon_psnr]
                    utils.plot_test_result(result_imgs, psnrs, img_num, save_dir=save_dir)

                    print('Test DB: %s, Saving result images...[%d/%d]' % (test_dataset, img_num, total_img_num))

            print('Test is finishied.')

    def test_single(self, img_fn):
        # networks
        self.model = Net(num_channels=self.num_channels, base_filter=64, num_residuals=16)

        if self.gpu_mode:
            self.model.cuda()

        # load model
        self.load_model()

        # load data
        img = Image.open(img_fn).convert('RGB')

        if self.num_channels == 1:
            img = img.convert('YCbCr')
            img_y, img_cb, img_cr = img.split()

            input = ToTensor()(img_y)
            y_ = Variable(input.unsqueeze(1))
        else:
            input = ToTensor()(img).view(1, -1, img.height, img.width)
            y_ = Variable(input)

        if self.gpu_mode:
            y_ = y_.cuda()

        # prediction
        self.model.eval()
        recon_img = self.model(y_)
        recon_img = recon_img.cpu().data[0].clamp(0, 1)
        recon_img = ToPILImage()(recon_img)

        if self.num_channels == 1:
            # merge color channels with super-resolved Y-channel
            recon_y = recon_img
            recon_cb = img_cb.resize(recon_y.size, Image.BICUBIC)
            recon_cr = img_cr.resize(recon_y.size, Image.BICUBIC)
            recon_img = Image.merge('YCbCr', [recon_y, recon_cb, recon_cr]).convert('RGB')

        # save img
        result_dir = os.path.join(self.save_dir, 'test_result')
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        save_fn = result_dir + '/SR_result.png'
        recon_img.save(save_fn)

        print('Single test result image is saved.')

    def save_model(self, epoch=None):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if epoch is not None:
            torch.save(self.model.state_dict(), model_dir + '/' + self.model_name +
                       '_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                       % (self.num_channels, self.batch_size, epoch, self.lr))
        else:
            torch.save(self.model.state_dict(), model_dir + '/' + self.model_name +
                       '_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                       % (self.num_channels, self.batch_size, self.num_epochs, self.lr))

        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.save_dir, 'model')

        model_name = model_dir + '/' + self.model_name +\
                     '_param_ch%d_batch%d_epoch%d_lr%.g.pkl'\
                     % (self.num_channels, self.batch_size, self.num_epochs, self.lr)
        if os.path.exists(model_name):
            self.model.load_state_dict(torch.load(model_name))
            print('Trained model is loaded.')
            return True
        else:
            print('No model exists to load.')
            return False
