from __future__ import absolute_import, division, print_function
from fileinput import filename

import os
import random
import numpy as np
import copy
from PIL import Image
import PIL.Image as pil

import torch
import torch.utils.data as data
from torchvision import transforms

a = np.array([[7.86605202e+03, 0.00000000e+00 ,1.38161169e+03 ,0.00000000e+00],
 [0.00000000e+00, 7.86605202e+03, 6.88564377e+02, 0.00000000e+00],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00]])

print(a[0])