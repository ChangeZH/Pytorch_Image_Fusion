import os
import cv2
import numpy as np
from PIL import Image
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class Fusion_Datasets(Dataset):
	"""docstring for Fusion_Datasets"""

	def __init__(self, configs, transform=None):
		super(Fusion_Datasets, self).__init__()
		self.root_dir = configs['root_dir']
		self.transform = transform
		self.channels = configs['channels']
		self.sensors = configs['sensors']
		self.img_list = {i: os.listdir(os.path.join(self.root_dir, i)) for i in self.sensors}
		self.img_path = {i: [os.path.join(self.root_dir, i, j) for j in os.listdir(os.path.join(self.root_dir, i))]
		                 for i in self.sensors}

	def __getitem__(self, index):
		img_data = {}
		for i in self.sensors:
			img = Image.open(self.img_path[i][index])
			# print(self.img_path[i][index])
			if self.channels == 1:
				img = img.convert('L')
			elif self.channels == 3:
				img = img.convert('RGB')
			if self.transform is not None:
				img = self.transform(img)
			img_data.update({i: img})
		return img_data

	def __len__(self):
		img_num = [len(self.img_list[i]) for i in self.img_list]
		img_counter = Counter(img_num)
		assert len(img_counter) == 1, 'Sensors Has Different length'
		return img_num[0]


if __name__ == '__main__':
	datasets = Fusion_Datasets(root_dir='../../datasets/TNO/', sensors=['Vis', 'Inf'],
	                           transform=transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor()]))
	train = DataLoader(datasets, 1, False)
	print(len(train))
	for i, data in enumerate(train):
		print(data)
