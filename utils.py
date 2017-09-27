import torch
from torch.autograd import Variable
from math import log10
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
# from edge_detector import edge_detect


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


# De-normalization
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# For logger
def to_np(x):
    return x.data.cpu().numpy()


def to_var(x):
    if torch.cuda.is_available():
        x = torch.from_numpy(x).cuda()
    return Variable(x)


# Plot losses
def plot_loss(avg_losses, num_epochs, save=False, save_dir='results/', show=False):
    fig, ax = plt.subplots()
    ax.set_xlim(0, num_epochs)
    temp = 0.0
    for i in range(len(avg_losses)):
        temp = max(np.max(avg_losses[i]), temp)
    ax.set_ylim(0, temp*1.1)
    plt.xlabel('# of Epochs')
    plt.ylabel('Loss values')

    plt.plot(avg_losses[0], label='G_loss')
    plt.plot(avg_losses[1], label='D_loss')
    plt.legend()

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_fn = 'Loss_values_epoch_{:d}'.format(num_epochs) + '.png'
        save_fn = os.path.join(save_dir, save_fn)
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


# Make gif
def make_gif(dataset, num_epochs, save_dir='results/'):
    gen_image_plots = []
    for epoch in range(num_epochs):
        # plot for generating gif
        save_fn = save_dir + 'Result_epoch_{:d}'.format(epoch + 1) + '.png'
        gen_image_plots.append(imageio.imread(save_fn))

    imageio.mimsave(save_dir + dataset + '_result_epochs_{:d}'.format(num_epochs) + '.gif', gen_image_plots, fps=5)


def weights_init_normal(m, mean=0.0, std=0.02):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def weights_init_kaming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv2d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('ConvTranspose2d') != -1:
        torch.nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        if m.bias is not None:
            m.bias.data.zero_()


def plot_test_result(imgs, psnrs, img_num, save=False, save_dir='results/', show=False, show_label=False):
    size = list(imgs[0].shape)
    if show_label:
        w = 9
        h = 3
    else:
        w = size[1] * 3 / 100
        h = size[2] / 100

    fig, axes = plt.subplots(1, 3, figsize=(w, h))
    # axes.axis('off')
    for i, (ax, img, psnr) in enumerate(zip(axes.flatten(), imgs, psnrs)):
        ax.axis('off')
        ax.set_adjustable('box-forced')

        if size[0] == 3:
            ax.imshow(img, cmap='gray', aspect='equal')
        else:
            img = img.squeeze()
            ax.imshow(img, cmap=None, aspect='equal')
        if show_label:
            ax.axis('on')
            if i == 0:
                ax.set_xlabel('HR image')
            elif i == 1:
                ax.set_xlabel('Bicubic (PSNR: %.2fdB)' % psnr)
            elif i == 2:
                ax.set_xlabel('SR image (PSNR: %.2fdB)' % psnr)

    if show_label:
        plt.tight_layout()
    else:
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.subplots_adjust(bottom=0)
        plt.subplots_adjust(top=1)
        plt.subplots_adjust(right=1)
        plt.subplots_adjust(left=0)

    # save figure
    if save:
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        save_fn = save_dir + '/Test_result_{:d}'.format(img_num) + '.png'
        plt.savefig(save_fn)

    if show:
        plt.show()
    else:
        plt.close()


def PSNR(pred, gt):
    diff = pred - gt
    mse = np.mean(diff ** 2)
    if mse == 0:
        return 100
    return 10 * log10(1.0 / mse)
