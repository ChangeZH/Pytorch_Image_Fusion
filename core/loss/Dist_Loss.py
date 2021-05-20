import torch
import torch.nn as nn


class Dist_Loss(nn.Module):
	"""docstring for Dist_Loss"""

	def __init__(self, sensors, p=2):
		super(Dist_Loss, self).__init__()
		self.p = p

	def forward(self, input_images, output_images):
		return sum([torch.dist(input_images[sensor], output_images[sensor], p=self.p) for sensor in input_images])
