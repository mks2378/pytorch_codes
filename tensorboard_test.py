# reference: https://pytorch.org/docs/stable/tensorboard.html
# reference: https://www.endtoend.ai/blog/pytorch-tensorboard/

# For tensorsboard usage practice

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])
trainset = datasets.MNIST('mnist_train', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
model = torchvision.models.resnet50(False)

# Have ResNet model take in gray-scale rather than RGB
model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
# dataiter = iter(trainloader)
# images, labels = next(dataiter)
# grid = torchvision.utils.make_grid(images)

"""
def writer.add_image(tag, img_tensor, global_step, ...):
Args:
    tag (string): Data identifier
    img_tensor (torch.Tensor, numpy.array, or string/blobname): Image data
    global_step (int): Global step value to record
"""

# writer.add_image('images', grid, 0)
# writer.add_graph(model, images)


"""
tensorboard.SummaryWriter will create 'runs' directory.
'runs' contains the tensorboard runs.

We can now run tensorboard. We need to specify where the runs are stored with '--logdir' flag.
$ tensorboard --logdir=runs (실험이 여러개라면 결과가 overwriting 되어 출력된다.)

특정 실험에 대한 log를 보고 싶다면, $tensorboard --logdir=runs/Aug10_10-28-21_mks

Visit 'https://localhost:6006/' with the browser of your choice. 
TensorBoard should appear with MNIST iamges

tensorboard 끄고 싶다면, command에서 Ctrl+C
"""

dataiter = iter(trainloader)
for n_iter in range(100):
    images, labels = next(dataiter)
    grid = torchvision.utils.make_grid(images)  # 여러 이미지를 하나로 합쳐서 출력
    writer.add_image('images', grid, n_iter)
    writer.add_graph(model, images)

    writer.add_scalar('Loss/train', np.random.random(), n_iter)
    writer.add_scalar('Loss/test', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
    writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    print(n_iter)

writer.close()

"""
def make_grid(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W) or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
    Add scalar data to summary.

    Args:
        tag (string): Data identifier
        scalar_value (float or string/blobname): Value to save
        global_step (int): Global step value to record
        walltime (float): Optional override default walltime (time.time())
        with seconds after epoch of event
"""