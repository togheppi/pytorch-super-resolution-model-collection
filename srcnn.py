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
            ConvBlock(num_channels, base_filter, 9, 1, 4, norm=None),
            ConvBlock(base_filter, base_filter // 2, 1, 1, 1, norm=None),
            ConvBlock(base_filter // 2, num_channels, 5, 1, 2, activation=None, norm=None),
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

    def load_dataset(self, dataset='train'):
        print('Loading datasets...')
        if dataset == 'train':
            train_set = get_training_set(self.data_dir, self.dataset, self.scale_factor, interpolation='bicubic')
            return DataLoader(dataset=train_set, num_workers=self.num_threads, batch_size=self.batch_size, shuffle=True)
        elif dataset == 'test':
            test_set = get_test_set(self.data_dir, self.dataset, self.scale_factor, interpolation='bicubic')
            return DataLoader(dataset=test_set, num_workers=self.num_threads, batch_size=self.test_batch_size, shuffle=False)
        
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
                self.optimizer.step()

                # log
                epoch_loss += loss.data[0]
                print("Epoch: [%2d] [%4d/%4d] loss: %.8f" % ((epoch + 1), (iter + 1), len(train_data_loader), loss.data[0]))

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
                utils.plot_test_result(result_imgs, psnrs, img_num, save=True, save_dir=result_dir, show_label=True)

                print("Saving %d test result images..." % img_num)

            psnr = utils.PSNR(recon_imgs, utils.to_np(x_))
            avg_psnr += psnr

        print("Avg. PSNR: {:.4f} dB".format(avg_psnr / len(test_data_loader)))

    def save_model(self, model_dir):
        torch.save(self.model.state_dict(), model_dir + '/' + self.model_name + '_param.pkl')

    def load_model(self, model_name):
        self.model.load_state_dict(torch.load(model_name))