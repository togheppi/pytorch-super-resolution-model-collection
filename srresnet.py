import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import models
import torch.utils.model_zoo as model_zoo
from base_networks import *
from torch.utils.data import DataLoader
from data import get_training_set, get_test_set
import utils
from logger import Logger


class Generator(torch.nn.Module):
    def __init__(self, num_channels, base_filter, num_residuals):
        super(Generator, self).__init__()

        self.input_conv = ConvBlock(num_channels, base_filter, 9, 1, 4, activation='lrelu', norm=None, bias=False)

        resnet_blocks = []
        for _ in range(num_residuals):
            resnet_blocks.append(ResnetBlock(base_filter, activation='lrelu', bias=False))
        self.residual_layers = nn.Sequential(*resnet_blocks)

        self.mid_conv = ConvBlock(base_filter, base_filter, 3, 1, 1, activation=None, bias=False)

        self.upscale4x = nn.Sequential(
            Upsample2xBlock(base_filter, base_filter, activation='lrelu', norm=None, bias=False),
            Upsample2xBlock(base_filter, base_filter, activation='lrelu', norm=None, bias=False)
        )

        self.output_conv = ConvBlock(base_filter, num_channels, 9, 1, 4, activation=None, norm=None, bias=False)

    def forward(self, x):
        out = self.input_conv(x)
        residual = out
        out = self.residual_layers(out)
        out = self.mid_conv(out)
        out = torch.add(out, residual)
        out = self.upscale4x(out)
        out = self.output_conv(out)
        return out

    def weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)


class Discriminator(torch.nn.Module):
    def __init__(self, num_channels, base_filter, image_size):
        super(Discriminator, self).__init__()
        self.image_size = image_size

        self.input_conv = ConvBlock(num_channels, base_filter, 3, 1, 1, activation='lrelu', norm=None, bias=False)

        self.conv_blocks = nn.Sequential(
            ConvBlock(base_filter, base_filter, 3, 2, 1, activation='lrelu', bias=False),
            ConvBlock(base_filter, base_filter * 2, 3, 1, 1, activation='lrelu', bias=False),
            ConvBlock(base_filter * 2, base_filter * 2, 3, 2, 1, activation='lrelu', bias=False),
            ConvBlock(base_filter * 2, base_filter * 4, 3, 1, 1, activation='lrelu', bias=False),
            ConvBlock(base_filter * 4, base_filter * 4, 3, 2, 1, activation='lrelu', bias=False),
            ConvBlock(base_filter * 4, base_filter * 8, 3, 1, 1, activation='lrelu', bias=False),
            ConvBlock(base_filter * 8, base_filter * 8, 3, 2, 1, activation='lrelu', bias=False),
        )

        self.dense_layers = nn.Sequential(
            DenseBlock(base_filter * 8 * image_size // 16 * image_size // 16, base_filter * 16, activation='lrelu', norm=None),
            DenseBlock(base_filter * 16, num_channels, activation='sigmoid', norm=None)
        )

    def forward(self, x):
        out = self.input_conv(x)
        out = self.conv_blocks(out)
        out = out.view(out.size()[0], -1)
        out = self.dense_layers(out)
        return out

    def weight_init(self, mean=0.0, std=0.001):
        for m in self.modules():
            utils.weights_init_normal(m, mean=mean, std=std)


class SRResNet(object):
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

        # networks
        self.model = Generator(num_channels=self.num_channels, base_filter=64, num_residuals=16)

        # weigh initialization
        self.model.weight_init()

        # optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))

        # loss function
        if self.gpu_mode:
            self.model.cuda()
            self.MSE_loss = nn.MSELoss().cuda()
        else:
            self.MSE_loss = nn.MSELoss()

        print('---------- Networks architecture -------------')
        utils.print_network(self.model)
        print('----------------------------------------------')

    def load_dataset(self, dataset='train'):
        print('Loading datasets...')
        if dataset == 'train':
            train_set = get_training_set(self.data_dir, self.dataset, self.scale_factor,
                                         interpolation='bicubic')
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)
        elif dataset == 'test':
            test_set = get_test_set(self.data_dir, self.dataset, self.scale_factor, interpolation='bicubic')
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False)

    def train(self):
        # load dataset
        train_data_loader = self.load_dataset(dataset='train')

        # set the logger
        log_dir = os.path.join(self.save_dir, 'logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = Logger(log_dir)

        # Train
        print('Training is started.')
        avg_loss = []
        step = 0
        self.model.train()
        for epoch in range(self.num_epochs):

            # learning rate is decayed by a factor of 10 every 20 epochs
            if (epoch + 1) % 20 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] /= 10.0
                print("Learning rate decay: lr={}".format(self.optimizer.param_groups[0]["lr"]))

            epoch_loss = 0
            for iter, (input, target) in enumerate(train_data_loader):
                # input data
                if self.gpu_mode:
                    x_ = Variable(target.cuda())
                    y_ = Variable(input.cuda())
                else:
                    x_ = Variable(target)
                    y_ = Variable(input)

                # update network
                self.optimizer.zero_grad()
                self.pred = self.model(y_)
                loss = self.MSE_loss(self.pred, x_)
                loss.backward()
                nn.utils.clip_grad_norm(self.model.parameters(), self.clip)
                self.optimizer.step()

                # log
                epoch_loss += loss.data[0]
                print("Epoch: [%2d] [%4d/%4d] loss: %.8f" % (
                    (epoch + 1), (iter + 1), len(train_data_loader), loss.data[0]))

                # tensorboard logging
                logger.scalar_summary('loss', loss.data[0], step + 1)
                step += 1

            # avg. loss per epoch
            avg_loss.append(epoch_loss / len(train_data_loader))

        # Plot avg. loss
        result_dir = os.path.join(self.save_dir, 'result')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        utils.plot_loss(avg_loss, self.num_epochs, save=True, save_dir=result_dir)
        print("Training is finished.")

        # Save trained parameters of model
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.save_model(model_dir)

    def test(self, model_name):
        # load model
        self.load_model(model_name)

        # load dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Test
        print('Test is started.')
        avg_psnr = 0
        img_num = 0
        self.model.eval()
        for input, target in test_data_loader:
            # input data
            if self.gpu_mode:
                x_ = Variable(target.cuda())
                y_ = Variable(input.cuda())
            else:
                x_ = Variable(target)
                y_ = Variable(input)

            # prediction
            self.pred = self.model(y_)
            recon_imgs = utils.to_np(self.pred)
            for i, recon_img in enumerate(recon_imgs):
                img_num += 1
                gt_img = utils.to_np(x_)
                bc_img = utils.to_np(y_)

                # calculate psnrs
                bc_psnr = utils.PSNR(bc_img, gt_img)
                recon_psnr = utils.PSNR(recon_img, gt_img)

                result_dir = os.path.join(self.save_dir, 'result')
                if not os.path.exists(result_dir):
                    os.mkdir(result_dir)

                # result_imgs = np.concatenate((gt_img, bc_img, recon_img), axis=1)
                # imsave(result_dir + '/Test_result_img_%d.png' % step, result_imgs)

                # save result images
                result_imgs = [gt_img, bc_img, recon_img]
                psnrs = [None, bc_psnr, recon_psnr]
                utils.plot_test_result(result_imgs, psnrs, img_num, save=True, save_dir=result_dir,
                                       show_label=True)

                print("Saving %d test result images..." % img_num)

            psnr = utils.PSNR(recon_imgs, utils.to_np(x_))
            avg_psnr += psnr

        print("Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_data_loader)))

    def save_model(self, model_dir):
        torch.save(self.model.state_dict(), model_dir + '/' + self.model_name + '_param.pkl')

    def load_model(self, model_name):
        self.model.load_state_dict(torch.load(model_name))


class FeatureExtractor(torch.nn.Module):
    def __init__(self, netVGG, feature_layer=8):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(netVGG.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


class SRGAN(object):
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
        print('Loading datasets...')
        if dataset == 'train':
            train_set = get_training_set(self.data_dir, self.dataset, self.scale_factor,
                                         interpolation='bicubic')
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size,
                              shuffle=True)
        elif dataset == 'test':
            test_set = get_test_set(self.data_dir, self.dataset, self.scale_factor, interpolation='bicubic')
            return DataLoader(dataset=test_set, num_workers=self.num_threads,
                              batch_size=self.test_batch_size,
                              shuffle=False)

    def train(self):
        # networks
        self.G = Generator(num_channels=self.num_channels, base_filter=64, num_residuals=16)
        self.D = Discriminator(num_channels=self.num_channels, base_filter=64, image_size=self.image_size)

        # weigh initialization
        self.G.weight_init()
        self.D.weight_init()

        # For the content loss
        self.feature_extractor = FeatureExtractor(models.vgg19(pretrained=True))

        # optimizer
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lr, betas=(0.9, 0.999))

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

        # load dataset
        train_data_loader = self.load_dataset(dataset='train')

        # set the logger
        G_log_dir = os.path.join(self.save_dir, 'G_logs')
        if not os.path.exists(G_log_dir):
            os.mkdir(G_log_dir)
        G_logger = Logger(G_log_dir)

        D_log_dir = os.path.join(self.save_dir, 'D_logs')
        if not os.path.exists(D_log_dir):
            os.mkdir(D_log_dir)
        D_logger = Logger(D_log_dir)

        img_log_dir = os.path.join(self.save_dir, 'img_logs')
        if not os.path.exists(img_log_dir):
            os.mkdir(img_log_dir)
        img_logger = Logger(img_log_dir)

        # Train
        print('Training is started.')
        # Avg. losses
        D_avg_loss = []
        G_avg_loss = []
        step = 0

        # test image for training
        test_input, test_target = train_data_loader.__iter__().__next__()

        self.D.train()
        for epoch in range(self.num_epochs):

            G_epoch_loss = 0
            D_epoch_loss = 0
            self.G.train()
            for iter, (input, target) in enumerate(train_data_loader):
                # input data
                mini_batch = target.size()[0]

                if self.gpu_mode:
                    x_ = Variable(target.cuda())
                    y_ = Variable(input.cuda())
                    # labels
                    real_label = Variable(torch.ones(mini_batch).cuda())
                    fake_label = Variable(torch.zeros(mini_batch).cuda())
                else:
                    x_ = Variable(target)
                    y_ = Variable(input)
                    # labels
                    real_label = Variable(torch.ones(mini_batch))
                    fake_label = Variable(torch.zeros(mini_batch))

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

                # Train generator
                self.G_optimizer.zero_grad()

                recon_image = self.G(y_)
                D_fake_decision = self.D(recon_image)

                # Adversarial loss
                GAN_loss = self.BCE_loss(D_fake_decision, real_label)

                # Content losses
                real_feature = self.feature_extractor(x_)
                fake_feature = self.feature_extractor(recon_image)
                content_loss = self.MSE_loss(real_feature, fake_feature)

                # Back propagation
                G_loss = 0.006 * content_loss + GAN_loss
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

            # generate fake images with fixed noise
            self.G.eval()
            test_recon_img = utils.to_np(self.G(Variable(test_input.cuda(), volatile=True)))
            gt_img = test_target.numpy()
            bc_img = test_input.numpy()

            # calculate psnrs
            bc_psnr = utils.PSNR(bc_img, gt_img)
            recon_psnr = utils.PSNR(test_recon_img, gt_img)

            result_dir = os.path.join(self.save_dir, 'result')
            if not os.path.exists(result_dir):
                os.mkdir(result_dir)

            # save result images
            result_imgs = [gt_img, bc_img, test_recon_img]
            psnrs = [None, bc_psnr, recon_psnr]
            utils.plot_test_result(result_imgs, psnrs, epoch, save=True, save_dir=result_dir,
                                   show_label=True)

            # log the images
            info = {
                'Reconstructed images': utils.to_np(test_recon_img).transpose(0, 2, 3, 1),  # convert to BxHxWxC
            }

            for tag, images in info.items():
                img_logger.image_summary(tag, images, epoch + 1)

        # Plot avg. loss
        # result_dir = os.path.join(self.save_dir, 'result')
        # if not os.path.exists(result_dir):
        #     os.mkdir(result_dir)
        # utils.plot_loss(avg_loss, self.num_epochs, save=True, save_dir=result_dir)
        print("Training is finished.")

        # Save trained parameters of model
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        self.save_model(model_dir)

    def test(self, model_name):
        # networks
        self.G = Generator(num_channels=self.num_channels, base_filter=64, num_residuals=16)
        if self.gpu_mode:
            self.G.cuda()

        # load model
        self.load_model(model_name)

        # load dataset
        test_data_loader = self.load_dataset(dataset='test')

        # Test
        print('Test is started.')
        avg_psnr = 0
        img_num = 0
        self.G.eval()
        for input, target in test_data_loader:
            # input data
            if self.gpu_mode:
                x_ = Variable(target.cuda())
                y_ = Variable(input.cuda())
            else:
                x_ = Variable(target)
                y_ = Variable(input)

            # prediction
            self.pred = self.G(y_)
            recon_imgs = utils.to_np(self.pred)
            for i, recon_img in enumerate(recon_imgs):
                img_num += 1
                gt_img = utils.to_np(x_)
                bc_img = utils.to_np(y_)

                # calculate psnrs
                bc_psnr = utils.PSNR(bc_img, gt_img)
                recon_psnr = utils.PSNR(recon_img, gt_img)

                result_dir = os.path.join(self.save_dir, 'result')
                if not os.path.exists(result_dir):
                    os.mkdir(result_dir)

                # result_imgs = np.concatenate((gt_img, bc_img, recon_img), axis=1)
                # imsave(result_dir + '/Test_result_img_%d.png' % step, result_imgs)

                # save result images
                result_imgs = [gt_img, bc_img, recon_img]
                psnrs = [None, bc_psnr, recon_psnr]
                utils.plot_test_result(result_imgs, psnrs, img_num, save=True, save_dir=result_dir,
                                       show_label=True)

                print("Saving %d test result images..." % img_num)

            psnr = utils.PSNR(recon_imgs, utils.to_np(x_))
            avg_psnr += psnr

        print("Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_data_loader)))

    def save_model(self, model_dir):
        torch.save(self.G.state_dict(), model_dir + '/' + self.model_name + '_G_param.pkl')
        torch.save(self.D.state_dict(), model_dir + '/' + self.model_name + '_D_param.pkl')

    def load_model(self, model_name):
        self.G.load_state_dict(torch.load(model_name))
