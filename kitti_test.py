import torch
import os
import numpy as np
import hickle as hkl

from torch.utils.data import DataLoader
from torch.autograd import Variable
from kitti_data import KITTI
from prednet import PredNet
from PIL import Image
import torchvision

def save_image(tensor, filename, nrow=8, padding=2,
               normalize=False, range=None, scale_each=False, pad_value=0):
    print(tensor.shape)
    im = Image.fromarray(np.uint8(tensor.numpy().transpose(1,2,0)*254.9))
    im.save(filename)
#from scipy.misc import imshow, imsave

batch_size = 16
A_channels = (3, 48, 96, 192)
R_channels = (3, 48, 96, 192)

DATA_DIR = 'kitti_data_raw'
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

nt = 10

kitti_test = KITTI(test_file, test_sources, nt)

test_loader = DataLoader(kitti_test, batch_size=batch_size, shuffle=False)

model = PredNet(R_channels, A_channels, output_mode='prediction')
model.load_state_dict(torch.load('models/training_0030.pt'))

if torch.cuda.is_available():
    print('Using GPU.')
    model.cuda()

for i, inputs in enumerate(test_loader):
    inputs = inputs.permute(0, 1, 4, 2, 3) # batch x time_steps x channel x width x height
    inputs = Variable(inputs.cuda())
    origin = inputs.data.cpu()[:, nt-1]
    print('origin:')
    print(type(origin))
    print(origin.size())

    print('predicted:')
    pred = model(inputs)
    pred = pred.data.cpu()
    print(type(pred))
    print(pred.size())
    origin = torchvision.utils.make_grid(origin, nrow=4)
    pred = torchvision.utils.make_grid(pred, nrow=4)
    save_image(origin, 'origin.jpg')
    save_image(pred, 'predicted.jpg')
    break

