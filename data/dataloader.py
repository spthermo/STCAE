import os

from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from utils import load_image, load_target_mask, load_target_hotspot

class SOR3DLoaderParams:
	def __init__(self, root_path):
		self.root_path = root_path

class SOR3DLoader(Dataset):
	def __init__(self, params):
		super(SOR3DLoader,self).__init__()
		self.params = params
		self.thres = 20

		root_path = self.params.root_path

		if not os.path.exists(root_path):
			raise ValueError("{} does not exist, exiting.".format(root_path))

		self.data = {}
		self.fdata = {}
		#Iterate over each recorded folder
		for recording in os.listdir(root_path):
			data_path = os.path.join(root_path,recording)
			if not os.path.isdir(data_path):
				continue

			self.data[recording] = {}
			self.fdata[recording] = {}
			#Data iteration
			splits = recording.split("_")
			self.data[recording]['frames'] = {}
			self.data[recording]['target'] = {}
			self.data[recording]['hotspot'] = {}
			self.data[recording]['label'] = {}
			self.data[recording]['label'] = splits[1]

			self.fdata[recording]['frames'] = {}
			self.fdata[recording]['target'] = {}
			self.fdata[recording]['hotspot'] = {}
			self.fdata[recording]['label'] = {}
			self.fdata[recording]['label'] = splits[1]

			num_files = sum([len(files) for r, d, files in os.walk(data_path + "\\rgb")])

			#new implementation
			num_files = num_files - 4
			step = 1

			if num_files > 20 and num_files <= 30:
				total_dummy_frames = 0
				dummy_counter = num_files - self.thres 
			elif num_files > 30 and num_files <= 40:
				total_dummy_frames = self.thres - (num_files // 2)
				step = 2
				dummy_counter = num_files % step
			elif num_files > 40 and num_files <= 60:
				total_dummy_frames = self.thres - (num_files // 3)
				step = 3
				dummy_counter = num_files % step
			elif num_files > 60 and num_files <= 80:
				total_dummy_frames = self.thres - (num_files // 4)
				step = 4
				dummy_counter = num_files % step
			elif num_files > 80 and num_files <= 100:
				total_dummy_frames = self.thres - (num_files // 5)
				step = 5
				dummy_counter = num_files % step
			else:
				total_dummy_frames = self.thres - (num_files // 6)
				step = 6
				dummy_counter = num_files % step

			if total_dummy_frames > 0:
				for i in range(total_dummy_frames):
					self.data[recording]['frames']["000" + str(i)] = {}
					self.data[recording]['frames']["000" + str(i)] = os.path.join(data_path + "\\rgb", "dummy")

			actual_frames = 0
			frame_counter = 1
			if dummy_counter == 0:
				next_frame = dummy_counter + step
			else:
				next_frame = 0
			for file in os.listdir(data_path + "\\rgb"):
				if frame_counter == dummy_counter:
					next_frame = frame_counter + step
				if frame_counter == next_frame and frame_counter <= num_files:
					actual_frames += 1
					next_frame = frame_counter + step
					file_split = file.split("_")
					full_filename = os.path.join(data_path + "\\rgb",file)
					if (len(file_split) < 2):
						self.data[recording]['frames'][file] = {}
						self.data[recording]['frames'][file] = full_filename
				frame_counter += 1
			cnt = 1
			for entry in self.data[recording]['frames']:
				if cnt % 2 == 0:
					self.fdata[recording]['frames'][entry] = {}
					self.fdata[recording]['frames'][entry] = self.data[recording]['frames'][entry]

				cnt += 1

			frame_counter = 1
			for file in os.listdir(data_path + "\\rgb"):
				full_filename = os.path.join(data_path + "\\rgb",file)
				if frame_counter == (num_files + 1):
					self.data[recording]['target'][file] = {}
					self.data[recording]['target'][file] = full_filename
					self.fdata[recording]['target'][file] = {}
					self.fdata[recording]['target'][file] = full_filename
					frame_counter += 1
				elif frame_counter == (num_files + 4):
					self.data[recording]['hotspot'][file] = {}
					self.data[recording]['hotspot'][file] = full_filename
					self.fdata[recording]['hotspot'][file] = {}
					self.fdata[recording]['hotspot'][file] = full_filename
					frame_counter += 1
				else:
					frame_counter += 1

				

			if (actual_frames + total_dummy_frames) < 20:
				inhere = 1
			if (actual_frames + total_dummy_frames) > 20:
				inhere = 2
		self.data.clear()
		self.__check__(self.fdata)

	def __check__(self, fdata):
		for recording in self.fdata:
			cnt = 0
			for frames in fdata[recording]['frames']:
				cnt +=1
			if cnt != 10:
				print(fdata[recording])

	def __len__(self):
		return len(self.fdata)


	def __getitem__(self, idx):
		key = list(self.fdata.keys())[idx]
		sample = self.fdata[key]
		
		vid = {}
		
		frame_cnt = 1
		for f in sample['frames']:
			frame = 'frame_' + '{:0>3}'.format(frame_cnt)
			color_img, action_class, object_class = load_image(sample['frames'][f])
			filename = sample['frames'][f]
			vid.update({
				frame : {
					"color" : color_img,
					"action" : action_class,
					"object" : object_class,
					"sequence_name" : filename
				}})
			frame_cnt += 1

		for f in sample['target']:
			target_img = load_target_mask(sample['target'][f], action_class)
			vid.update({
				frame : {
					"color" : color_img,
					"action" : action_class,
					"object" : object_class,
					"sequence_name" : filename,
					"target" : target_img
				}})

		for f in sample['hotspot']:
			hotspot_img = load_target_hotspot(sample['hotspot'][f])
			vid.update({
				frame : {
					"color" : color_img,
					"action" : action_class,
					"object" : object_class,
					"sequence_name" : filename,
					"target" : target_img,
					"heatmap" : hotspot_img
				}})

		return vid