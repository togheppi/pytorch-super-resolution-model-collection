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
    def __init__(self, num_channels, base_filter, num_recursions):
        super(Net, self).__init__()
        self.num_recursions = num_recursions
        # embedding layer
        self.embedding_layer = nn.Sequential(
            ConvBlock(num_channels, base_filter, 3, 1, 1, norm=None),
            ConvBlock(base_filter, base_filter, 3, 1, 1, norm=None)
        )

        # conv block of inference layer
        self.conv_block = ConvBlock(base_filter, base_filter, 3, 1, 1, norm=None)

        # reconstruction layer
        self.reconstruction_layer = nn.Sequential(
            ConvBlock(base_filter, base_filter, 3, 1, 1, activation=None, norm=None),
            ConvBlock(base_filter, num_channels, 3, 1, 1, activation=None, norm=None)
        )

        # initial w
        self.w_init = torch.ones(self.num_recursions) / self.num_recursions
        self.w = Variable(self.w_init.cuda(), requires_grad=True)

    def forward(self, x):
        # embedding layer
        h0 = self.embedding_layer(x)

        # recursions
        h = [h0]
        for d in range(self.num_recursions):
            h.append(self.conv_block(h[d]))

        y_d_ = []
        out_sum = 0
        for d in range(self.num_recursions):
            y_d_.append(self.reconstruction_layer(h[d+1]))
            out_sum += torch.mul(y_d_[d], self.w[d])
        out_sum = torch.mul(out_sum, 1.0 / (torch.sum(self.w)))

        # skip connection
        final_out = torch.add(out_sum, x)

        return y_d_, final_out

    def weight_init(self):
        for m in self.modules():
            utils.weights_init_kaming(m)


class DRCN(object):
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
        self.num_recursions = 16
        self.model = Net(num_channels=self.num_channels, base_filter=256, num_recursions=self.num_recursions)

        # weigh initialization
        self.model.weight_init()

        # optimizer
        self.momentum = 0.9
        self.weight_decay = 0.0001
        self.loss_alpha = 1.0
        self.loss_alpha_zero_epoch = 25
        self.loss_alpha_decay = self.loss_alpha / self.loss_alpha_zero_epoch
        self.loss_beta = 0.001

        # learnable parameters
        param_groups = list(self.model.parameters())
        param_groups = [{'params': param_groups}]
        param_groups += [{'params': [self.model.w]}]
        self.optimizer = optim.Adam(param_groups, lr=self.lr)

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
        test_input, test_target = test_data_loader.dataset.__getitem__(2)
        test_input = test_input.unsqueeze(0)
        test_target = test_target.unsqueeze(0)

        self.model.train()
        for epoch in range(self.num_epochs):

            # learning rate is decayed by a factor of 10 every 20 epochs
            if (epoch + 1) % 20 == 0:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] /= 10.0
                print("Learning rate decay: lr={}".format(self.optimizer.param_groups[0]["lr"]))

            # loss_alpha decayed to zero after 25 epochs
            self.loss_alpha = max(0.0, self.loss_alpha - self.loss_alpha_decay)

            epoch_loss = 0
            for iter, (input, target) in enumerate(train_data_loader):
                # input data (bicubic interpolated image)
                if self.gpu_mode:
                    y = Variable(target.cuda())
                    x = Variable(utils.img_interp(input, self.scale_factor).cuda())
                else:
                    y = Variable(target)
                    x = Variable(utils.img_interp(input, self.scale_factor))

                # update network
                self.optimizer.zero_grad()
                y_d_, y_ = self.model(x)

                # loss1
                loss1 = 0
                for d in range(self.num_recursions):
                    loss1 += (self.MSE_loss(y_d_[d], y) / self.num_recursions)

                # loss2
                loss2 = self.MSE_loss(y_, y)

                # regularization
                reg_term = 0
                for theta in self.model.parameters():
                    reg_term += torch.mean(torch.sum(theta ** 2))

                # total loss

                loss = self.loss_alpha * loss1 + (1-self.loss_alpha) * loss2 + self.loss_beta * reg_term
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
            _, recon_imgs = self.model(Variable(utils.img_interp(test_input, self.scale_factor).cuda()))
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
        self.num_recursions = 16
        self.model = Net(num_channels=self.num_channels, base_filter=256, num_recursions=self.num_recursions)

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
                y_ = Variable(utils.img_interp(input, self.scale_factor).cuda())
            else:
                y_ = Variable(utils.img_interp(input, self.scale_factor))

            # prediction
            _, recon_imgs = self.model(y_)
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

    def test_single(self, img_fn):
        # networks
        self.model = Net(num_channels=self.num_channels, base_filter=256, num_recursions=self.num_recursions)

        if self.gpu_mode:
            self.model.cuda()

        # load model
        self.load_model()

        # load data
        img = Image.open(img_fn)
        img = img.convert('YCbCr')
        y, cb, cr = img.split()

        input = Variable(ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
        if self.gpu_mode:
            input = input.cuda()

        self.model.eval()
        recon_img = self.model(input)

        # save result images
        utils.save_img(recon_img.cpu().data, 1, save_dir=self.save_dir)

        out = recon_img.cpu()
        out_img_y = out.data[0]
        out_img_y = (((out_img_y - out_img_y.min()) * 255) / (out_img_y.max() - out_img_y.min())).numpy()
        # out_img_y *= 255.0
        # out_img_y = out_img_y.clip(0, 255)
        out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

        out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
        out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
        out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

        # save img
        result_dir = os.path.join(self.save_dir, 'result')
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        save_fn = result_dir + '/SR_result.png'
        out_img.save(save_fn)

    def save_model(self, epoch=None):
        model_dir = os.path.join(self.save_dir, 'model')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if epoch is not None:
            torch.save(self.model.state_dict(), model_dir + '/' + self.model_name + '_param_epoch_%d.pkl' % epoch)
            torch.save(self.model.w, model_dir + '/' + self.model_name + '_w_epoch_%d.pkl' % epoch)
        else:
            torch.save(self.model.state_dict(), model_dir + '/' + self.model_name + '_param.pkl')
            torch.save(self.model.w, model_dir + '/' + self.model_name + '_w.pkl')

        print('Trained model is saved.')

    def load_model(self):
        model_dir = os.path.join(self.save_dir, 'model')

        model_name = model_dir + '/' + self.model_name + '_param.pkl'
        if os.path.exists(model_name):
            self.model.load_state_dict(torch.load(model_name))
            print('Trained model is loaded.')
        else:
            print('No model exists to load.')

        w_name = model_dir + '/' + self.model_name + '_w.pkl'
        if os.path.exists(w_name):
            self.model.w = torch.load(w_name)
            print('Trained weight is loaded.')
        else:
            print('No weight exists to load.')
