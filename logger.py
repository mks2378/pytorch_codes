# from logger import GANLogger
"""
<train.py>
from logger import GANLogger

# Log Progress
if i%50 == 0:
    logger.log_train_image(img_grid, epoch)
logger.log_training(loss_D.item(), loss_G.item(), loss_content.item(), loss_pixel.item(), loss_tri.item(), t_iteration)
...
img_lr = make_grid(img_lr, nrow=opt.batch_size//4, normalize=True)
img_hr = make_grid(img_hr, nrow=opt.batch_size//4, normalize=True)
gen_hr = make_grid(gen_hr, nrow=opt.batch_size//4, normalize=True)
...
img_grid = torch.cat((img_hr, gen_hr), 1)
...
logger.log_validation(loss_G.item(), generator, img_lr, img_grid, epoch, t_iteration)
"""

import random
import torch
from torch.utils.tensorboard import SummaryWriter

class GANLogger(SummaryWriter):
    def __init__(self, logdir):
        super(GANLogger, self).__init__(logdir)

    def log_training(self, D_loss, adv_loss, content_loss, pixel_loss, iteration):
        self.add_scalar("Critic.t_loss", D_loss, iteration)
        self.add_scalar("Adversarial.t_loss", adv_loss, iteration)
        self.add_scalar("Content.t_loss", content_loss, iteration)
        self.add_scalar("pixel.t_loss", pixel_loss, iteration)

    def log_train_image(self, img_grid, epoch):
        self.add_image("generated_train", img_grid, epoch)

    def log_sampling_image(self, hr_img_grid, epoch):
        self.add_scalar("aligned image", hr_img_grid, epoch)

    def log_validation(self, psnr, ssim, model, img_lr, img_grid, epoch, iteration):
        self.add_scalar("psnr", psnr, iteration)
        self.add_scalar("ssim", ssim, iteration)
        """
        for tag, value in model.named_parameters():
            tag = tag.replace('.', '/')
            self.add_histogram(tag, value.data.cpu().numpy(), iteration)
        """
        self.add_image("input_val", img_lr, epoch)
        self.add_image("generated_val", img_grid, epoch)
