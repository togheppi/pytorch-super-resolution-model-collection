import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
from base_networks import *
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import utils
from logger import Logger
from torchvision.transforms import *


class Generator(torch.nn.Module):
    def __init__(self, num_channels, base_filter, num_residuals):
        super(Generator, self).__init__()

        self.input_conv = ConvBlock(num_channels, base_filter, 9, 1, 4, activation='prelu', norm=None)

        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResnetBlock(base_filter, activation='prelu'))
        self.residual_layers = nn.Sequential(*resnet_blocks)

        self.mid_conv = ConvBlock(base_filter, base_filter, 3, 1, 1, activation=None)

        self.upscale4x = nn.Sequential(
            Upsample2xBlock(base_filter, base_filter, upsample='ps', activation='prelu', norm=None),
            Upsample2xBlock(base_filter, base_filter, upsample='ps', activation='prelu', norm=None)
        )

        self.output_conv = ConvBlock(base_filter, num_channels, 9, 1, 4, activation=None, norm=None)

    def forward(self, x):
        out = self.input_conv(x)
        residual = out
        out = self.residual_layers(out)
        out = self.mid_conv(out)
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.output_conv(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)


class Discriminator(torch.nn.Module):
    def __init__(self, num_channels, base_filter, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='lrelu', norm=None)

        self.conv_blocks = nn.Sequential(
            ConvBlock(base_filter, base_filter, 3, 2, 1, activation='lrelu'),
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation='lrelu'),
            ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation='lrelu'),
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation='lrelu'),
            ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation='lrelu'),
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation='lrelu'),
            ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation='lrelu'),
        )

        self.dense_layers = nn.Sequential(
            DenseBlock(base_filter * 8 * image_size // 16 * image_size // 16, base_filter * 16, activation='lrelu',
                       norm=None),
            DenseBlock(base_filter * 16, 1, activation='sigmoid', norm=None)
        )

    def forward(self, x):
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        out = out.view(out.size()[0], -1)
        out = self.dense_layers(out)
        return out

    def weight_init(self, mean=0.0, std=0.02):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)


class FeatureExtractor(torch.nn.Module):
    def __init__(self, netVGG, feature_layer=8):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(netVGG.features.children())[:(feature_layer + 1)])

    def forward(self, x):
        return self.features(x)


class SRGAN(object):
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
        # load dataset
        train_data_loader = self.load_dataset(dataset=self.train_dataset, is_train=True)
        test_data_loader = self.load_dataset(dataset=self.test_dataset[0], is_train=False)

        # networks
        self.G = Generator(num_channels=self.num_channels, base_filter=64, num_residuals=16)
        self.D = Discriminator(num_channels=self.num_channels, base_filter=64, image_size=self.crop_size)

        # weigh initialization
        self.G.weight_init()
        self.D.weight_init()

        # For the content loss
        self.feature_extractor = FeatureExtractor(models.vgg19(pretrained=True))

        # optimizer
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.9, 0.999))
        # self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.D_optimizer = optim.SGD(self.D.parameters(), lr=self.lr/100, momentum=0.9, nesterov=True)

        # loss function
        if self.gpu_mode:
            self.G.cuda()
            self.D.cuda()
            self.feature_extractor.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
            self.BCE_loss = nn.BCELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()
            self.BCE_loss = nn.BCELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('----------------------------------------------')

        # set the logger
        G_log_dir = os.path.join(self.save_dir, 'G_logs')
        if not os.path.exists(G_log_dir):
            os.mkdir(G_log_dir)
        G_logger = Logger(G_log_dir)

        D_log_dir = os.path.join(self.save_dir, 'D_logs')
        if not os.path.exists(D_log_dir):
            os.mkdir(D_log_dir)
        D_logger = Logger(D_log_dir)

        ################# Pre-train generator #################
        self.epoch_pretrain = 50

        # Load pre-trained parameters of generator
        if not self.load_model(is_pretrain=True):
            # Pre-training generator for 50 epochs
            print('Pre-training is started.')
            self.G.train()
            for epoch in range(self.epoch_pretrain):
                for iter, (lr, hr, _) in enumerate(train_data_loader):
                    # input data (low resolution image)
                    if self.num_channels == 1:
                        x_ = Variable(utils.norm(hr[:, 0].unsqueeze(1), vgg=True))
                        y_ = Variable(utils.norm(lr[:, 0].unsqueeze(1), vgg=True))
                    else:
                        x_ = Variable(utils.norm(hr, vgg=True))
                        y_ = Variable(utils.norm(lr, vgg=True))

                    if self.gpu_mode:
                        x_ = x_.cuda()
                        y_ = y_.cuda()

                    # Train generator
                    self.G_optimizer.zero_grad()
                    recon_image = self.G(y_)

                    # Content losses
                    content_loss = self.MSE_loss(recon_image, x_)

                    # Back propagation
                    G_loss_pretrain = content_loss
                    G_loss_pretrain.backward()
                    self.G_optimizer.step()

                    # log
                    print("Epoch: [%2d] [%4d/%4d] G_loss_pretrain: %.8f"
                          % ((epoch + 1), (iter + 1), len(train_data_loader), G_loss_pretrain.data[0]))

            print('Pre-training is finished.')

            # Save pre-trained parameters of generator
            self.save_model(is_pretrain=True)

        ################# Adversarial train #################
        print('Training is started.')
        # Avg. losses
        G_avg_loss = []
        D_avg_loss = []
        step = 0

        # test image
        test_lr, test_hr, test_bc = test_data_loader.dataset.__getitem__(2)
        test_lr = test_lr.unsqueeze(0)
        test_hr = test_hr.unsqueeze(0)
        test_bc = test_bc.unsqueeze(0)

        self.G.train()
        self.D.train()
        for epoch in range(self.num_epochs):

            # learning rate is decayed by a factor of 10 every 20 epoch
            if (epoch + 1) % 20 == 0:
                for param_group in self.G_optimizer.param_groups:
                    param_group["lr"] /= 10.0
                print("Learning rate decay for G: lr={}".format(self.G_optimizer.param_groups[0]["lr"]))
                for param_group in self.D_optimizer.param_groups:
                    param_group["lr"] /= 10.0
                print("Learning rate decay for D: lr={}".format(self.D_optimizer.param_groups[0]["lr"]))

            G_epoch_loss = 0
            D_epoch_loss = 0
            for iter, (lr, hr, _) in enumerate(train_data_loader):
                # input data (low resolution image)
                mini_batch = lr.size()[0]

                if self.num_channels == 1:
                    x_ = Variable(utils.norm(hr[:, 0].unsqueeze(1), vgg=True))
                    y_ = Variable(utils.norm(lr[:, 0].unsqueeze(1), vgg=True))
                else:
                    x_ = Variable(utils.norm(hr, vgg=True))
                    y_ = Variable(utils.norm(lr, vgg=True))

                if self.gpu_mode:
                    x_ = x_.cuda()
                    y_ = y_.cuda()
                    # labels
                    real_label = Variable(torch.ones(mini_batch).cuda())
                    fake_label = Variable(torch.zeros(mini_batch).cuda())
                else:
                    # labels
                    real_label = Variable(torch.ones(mini_batch))
                    fake_label = Variable(torch.zeros(mini_batch))

                # Reset gradient
                self.D_optimizer.zero_grad()

                # Train discriminator with real data
                D_real_decision = self.D(x_)
                D_real_loss = self.BCE_loss(D_real_decision, real_label)

                # Train discriminator with fake data
                recon_image = self.G(y_)
                D_fake_decision = self.D(recon_image)
                D_fake_loss = self.BCE_loss(D_fake_decision, fake_label)

                D_loss = D_real_loss + D_fake_loss

                # Back propagation
                D_loss.backward()
                self.D_optimizer.step()

                # Reset gradient
                self.G_optimizer.zero_grad()

                # Train generator
                recon_image = self.G(y_)
                D_fake_decision = self.D(recon_image)

                # Adversarial loss
                GAN_loss = self.BCE_loss(D_fake_decision, real_label)

                # Content losses
                mse_loss = self.MSE_loss(recon_image, x_)
                x_VGG = Variable(utils.norm(hr, vgg=True).cuda())
                recon_VGG = Variable(utils.norm(recon_image.data, vgg=True).cuda())
                real_feature = self.feature_extractor(x_VGG)
                fake_feature = self.feature_extractor(recon_VGG)
                vgg_loss = self.MSE_loss(fake_feature, real_feature.detach())

                # Back propagation
                G_loss = mse_loss + 6e-3 * vgg_loss + 1e-3 * GAN_loss
                G_loss.backward()
                self.G_optimizer.step()

                # log
                G_epoch_loss += G_loss.data[0]
                D_epoch_loss += D_loss.data[0]
                print("Epoch: [%2d] [%4d/%4d] G_loss: %.8f, D_loss: %.8f"
                      % ((epoch + 1), (iter + 1), len(train_data_loader), G_loss.data[0], D_loss.data[0]))

                # tensorboard logging
                G_logger.scalar_summary('losses', G_loss.data[0], step + 1)
                D_logger.scalar_summary('losses', D_loss.data[0], step + 1)
                step += 1

            # avg. loss per epoch
            G_avg_loss.append(G_epoch_loss / len(train_data_loader))
            D_avg_loss.append(D_epoch_loss / len(train_data_loader))

            # prediction
            if self.num_channels == 1:
                y_ = Variable(utils.norm(test_lr[:, 0].unsqueeze(1), vgg=True))
            else:
                y_ = Variable(utils.norm(test_lr, vgg=True))

            if self.gpu_mode:
                y_ = y_.cuda()

            recon_img = self.G(y_)
            sr_img = utils.denorm(recon_img[0].cpu().data, vgg=True)

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
        utils.plot_loss([G_avg_loss, D_avg_loss], self.num_epochs, save_dir=self.save_dir)
        print("Training is finished.")

        # Save final trained parameters of model
        self.save_model(epoch=None)

    def test(self):
        # networks
        self.G = Generator(num_channels=self.num_channels, base_filter=64, num_residuals=16)

        if self.gpu_mode:
            self.G.cuda()

        # load model
        self.load_model()

        # load dataset
        for test_dataset in self.test_dataset:
            test_data_loader = self.load_dataset(dataset=test_dataset, is_train=False)

            # Test
            print('Test is started.')
            img_num = 0
            total_img_num = len(test_data_loader)
            self.G.eval()
            for lr, hr, bc in test_data_loader:
                # input data (low resolution image)
                if self.num_channels == 1:
                    y_ = Variable(utils.norm(lr[:, 0].unsqueeze(1), vgg=True))
                else:
                    y_ = Variable(utils.norm(lr, vgg=True))

                if self.gpu_mode:
                    y_ = y_.cuda()

                # prediction
                recon_imgs = self.G(y_)
                for i, recon_img in enumerate(recon_imgs):
                    img_num += 1
                    sr_img = utils.denorm(recon_img.cpu().data, vgg=True)

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
        self.G = Generator(num_channels=self.num_channels, base_filter=64, num_residuals=16)

        if self.gpu_mode:
            self.G.cuda()

        # load model
        self.load_model()

        # load data
        img = Image.open(img_fn).convert('RGB')

        if self.num_channels == 1:
            img = img.convert('YCbCr')
            img_y, img_cb, img_cr = img.split()

            input = ToTensor()(img_y)
            y_ = Variable(utils.norm(input.unsqueeze(1), vgg=True))
        else:
            input = ToTensor()(img).view(1, -1, img.height, img.width)
            y_ = Variable(utils.norm(input, vgg=True))

        if self.gpu_mode:
            y_ = y_.cuda()

        # prediction
        self.G.eval()
        recon_img = self.G(y_)
        recon_img = utils.denorm(recon_img.cpu().data[0].clamp(0, 1), vgg=True)
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

    def save_model(self, epoch=None, is_pretrain=False):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        if is_pretrain:
            torch.save(self.G.state_dict(), model_dir + '/' + self.model_name + '_G_param_pretrain.pkl')
            print('Pre-trained generator model is saved.')
        else:
            if epoch is not None:
                torch.save(self.G.state_dict(), model_dir + '/' + self.model_name +
                           '_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, epoch, self.lr))
                torch.save(self.D.state_dict(), model_dir + '/' + self.model_name +
                           '_D_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, epoch, self.lr))
            else:
                torch.save(self.G.state_dict(), model_dir + '/' + self.model_name +
                           '_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, self.num_epochs, self.lr))
                torch.save(self.D.state_dict(), model_dir + '/' + self.model_name +
                           '_D_param_ch%d_batch%d_epoch%d_lr%.g.pkl'
                           % (self.num_channels, self.batch_size, self.num_epochs, self.lr))
            print('Trained models are saved.')

    def load_model(self, is_pretrain=False):
        model_dir = os.path.join(self.save_dir, 'model')

        if is_pretrain:
            model_name = model_dir + '/' + self.model_name + '_G_param_pretrain.pkl'
            if os.path.exists(model_name):
                self.G.load_state_dict(torch.load(model_name))
                print('Pre-trained generator model is loaded.')
                return True
        else:
            model_name = model_dir + '/' + self.model_name + \
                         '_G_param_ch%d_batch%d_epoch%d_lr%.g.pkl' \
                         % (self.num_channels, self.batch_size, self.num_epochs, self.lr)
            if os.path.exists(model_name):
                self.G.load_state_dict(torch.load(model_name))
                print('Trained generator model is loaded.')
                return True

        return False


