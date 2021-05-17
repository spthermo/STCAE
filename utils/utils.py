import torch
import torch.nn as nn
import cv2
import numpy
import json
import math
from PIL import Image

def get_action_class_id(x):
	if x == 'cut':
		id = 0
	elif x == 'grasp':
		id = 1
	elif x == 'hammer':
		id = 2
	elif x == 'lift':
		id = 3
	elif x == 'paint':
		id = 4
	elif x == 'push':
		id = 5
	elif x == 'rotate':
		id = 6
	elif x == 'squeeze':
		id = 7
	elif x == 'type':
		id = 8
	else:
		id = 9
	
	return id


def get_object_class_id(x):
	if x == 'ball':
		id = 0
	elif x == 'book':
		id = 1
	elif x == 'bottle':
		id = 2
	elif x == 'brush':
		id = 3
	elif x == 'can':
		id = 4
	elif x == 'cup':
		id = 5
	elif x == 'hammer':
		id = 6
	elif x == 'knife':
		id = 7
	elif x == 'pitcher':
		id = 8
	elif x == 'smartphone':
		id = 9
	elif x == 'sponge':
		id = 10
	else:
		id = 100
	
	return id


def get_channel_values(id):
	if id == 0:
		b = 0; g = 165; r = 255; c = 1
	elif id == 1:
		b = 0; g = 255; r = 0; c = 2
	elif id == 2:
		b = 100; g = 255; r = 0; c = 3
	elif id == 3:
		b = 40; g = 255; r = 0; c = 4
	elif id == 4:
		b = 200; g = 255; r = 0; c = 5
	elif id == 5:
		b = 240; g = 255; r = 120; c = 6
	elif id == 6:
		b = 60; g = 255; r = 0; c = 7
	elif id == 7:
		b = 160; g = 255; r = 0; c = 8
	elif id == 8:
		b = 240; g = 255; r = 0; c = 9
	else:
		b = 0; g = 0; r = 0; c = 0

	return b, g, r, c


def load_image(filename, data_type=torch.float32):
	views_id_path = "misc\\all_views_id.txt"
	view = 0
	filename_split = filename.split("\\")[-1]
	sample_name = filename.split("\\")[-3]
	subject_name = sample_name.split("_")[0]
	with open(views_id_path, 'r') as f:
		for cnt, line in enumerate(f):
			subject = line.split("\t")[0]
			kinect_id = line.split("\t")[1]
			if subject == subject_name:
				view = int(kinect_id)
				break

	action_class = sample_name.split("_")[1]
	aclass = get_action_class_id(action_class)

	object_class = sample_name.split("_")[-1]
	oclass = get_object_class_id(object_class)

	if filename_split == "dummy":
		return load_zero_image(), aclass, oclass
	else:
		color_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYCOLOR))
		h_in, w_in, c_in = color_img.shape
		resized_image = cv2.resize(color_img, (640, 360))
		h, w, c = resized_image.shape
		if view == 1:
			crop_img = resized_image[0:300, 200:500]
		else:
			crop_img = resized_image[0:300, 200:500]
		h, w, c = crop_img.shape
		color_data = crop_img.astype(numpy.float32).transpose(2, 0, 1)

		return torch.from_numpy(color_data).type(data_type) / 255.0, aclass, oclass


def load_target_mask(filename, class_id, data_type=torch.float32):
	target = torch.zeros(300, 300)
	views_id_path = "misc\\all_views_id.txt"
	view = 0
	filename_split = filename.split("\\")[-1]
	sample_name = filename.split("\\")[-3]
	subject_name = sample_name.split("_")[0]
	with open(views_id_path, 'r') as f:
		for cnt, line in enumerate(f):
			subject = line.split("\t")[0]
			kinect_id = line.split("\t")[1]
			if subject == subject_name:
				view = int(kinect_id)
				break

	color_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYCOLOR))
	h_in, w_in, c_in = color_img.shape
	resized_image = cv2.resize(color_img, (640, 360))
	h, w, c = resized_image.shape
	if view == 1:
		crop_img = resized_image[0:300, 200:500]
	else:
		crop_img = resized_image[0:300, 200:500]

	h, w, c = crop_img.shape
	color_data = crop_img.astype(numpy.float32).transpose(2, 0, 1)
	tensor_from_numpy = torch.from_numpy(color_data.astype(numpy.int))

	Blue, Green, Red, class_num = get_channel_values(class_id)

	for i in range(tensor_from_numpy.shape[1]):
		for j in range(tensor_from_numpy.shape[1]):
			if tensor_from_numpy[0][i][j] == Blue and tensor_from_numpy[1][i][j] == Green and tensor_from_numpy[2][i][j] == Red:
				target[i][j] = class_id
			else:
				target[i][j] = 9

	return target.long()

def load_target_hotspot(filename):
	filename_split = filename.split("\\")[-1]
	with open(filename) as json_file:
		data = json.load(json_file)
		x_c = int(data['center'][0]['x_coord'])
		y_c = int(data['center'][0]['y_coord'])
		radius = data['center'][0]['radius']

	blank_img = numpy.zeros((1080, 1920, 3), numpy.uint8)
	blank_img[:, :] = [0, 0, 0]
	cv2.circle(blank_img, (x_c, y_c), radius, (0, 0, 255), 10)
	
	resized_image = cv2.resize(blank_img, (640, 360))
	crop_img = resized_image[0:300, 200:500]
	crop_img [crop_img == 255] = 1
	heatmap = crop_img[:,:,2]
	tensor_from_numpy = torch.from_numpy(heatmap.astype(numpy.int))

	return tensor_from_numpy.float()

def load_zero_image():
	return torch.zeros(3, 300, 300)

def generate_gt_heatmap(target, kernel_size, sigma=3):
    mask = target.clone()

    mask = mask.unsqueeze(1).float()
    heatmap = mask.clone()

    x_coord = torch.arange(kernel_size).float()
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size).float()
    y_grid = x_grid.t().float()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                      torch.exp(
                          -torch.sum((xy_grid - mean)**2., dim=-1) /\
                          (2*variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    #conv layer will be used for Gaussian blurring
    gconv = nn.Conv2d(1, 1, kernel_size, stride=1, padding=2, bias=False)

    #init kernels with Gaussian distribution
    gconv.weight.data = gaussian_kernel
    gconv.weight.requires_grad = False

    for i in range(mask.shape[0]):
        temp = mask[i].clone()
        y = gconv(temp.unsqueeze(0))
        heatmap[i] = y.squeeze(0)

    heatmap[heatmap != 0] = 1
    
    return heatmap.squeeze(1)