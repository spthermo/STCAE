import os
import sys
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import cv2
import numpy


def load_image(filename, data_type=torch.float32):
	color_img = numpy.array(cv2.imread(filename, cv2.IMREAD_ANYCOLOR))
	crop_img = color_img[0:1080, 420:1500, 0:3]
	resized_image = cv2.resize(crop_img, (512, 512))
	h, w, c = resized_image.shape
	color_data = resized_image.astype(numpy.float32).transpose(2, 0, 1)

	return torch.from_numpy(color_data).type(data_type) / 255.0


def load_zero_image():
	return torch.zeros(3, 512, 512)


class data_loader_params:
	def __init__(self, root_path):
		self.root_path = root_path


class data_loader(Dataset):
	def __init__(self, params):
		super(data_loader,self).__init__()
		self.params = params

		root_path = self.params.root_path

		if not os.path.exists(root_path):
			raise ValueError("{} does not exist, exiting.".format(root_path))

		self.data = {}

		#Iterate over each recorded folder
		for recording in os.listdir(root_path):
			data_path = os.path.join(root_path,recording)
			if not os.path.isdir(data_path):
				continue

			self.data[recording] = {}
			#Data iteration
			splits = recording.split("_")
			self.data[recording]['frames'] = {}
			self.data[recording]['target'] = {}
			self.data[recording]['label'] = {}
			self.data[recording]['label'] = splits[1]
			for file in os.listdir(data_path):
				file_split = file.split("_")
				full_filename = os.path.join(data_path,file)
				if (len(file_split) < 2):
					self.data[recording]['frames'][file] = {}
					self.data[recording]['frames'][file]= full_filename
				else:
					if (file_split[1] == 'color'):
						self.data[recording]['target'][file] = {}
						self.data[recording]['target'][file]= full_filename


	def __len__(self):
		return len(self.data)


	def __getitem__(self, idx):
		key = list(self.data.keys())[idx]
		sample = self.data[key]
		
		vid = {}
		
		num_frames = len(sample['frames'])
		
		for i in range(81-num_frames):
			frame = 'frame_' '{:0>3}'.format(i)
			color_img = load_zero_image()
			vid.update({
				frame : {
					"color" : color_img
				}})
		
		index = 81-num_frames
		for f in sample['frames']:
			frame = 'frame_' + '{:0>3}'.format(index)
			color_img = load_image(sample['frames'][f])
			vid.update({
				frame : {
					"color" : color_img
				}})
			index += 1

		return vid