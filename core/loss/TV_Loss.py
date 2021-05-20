import torch
import torch.nn as nn


class TV_Loss(nn.Module):
	"""docstring for TV_Loss"""

	def __init__(self, sensors, num_inputs=2):
		super(TV_Loss, self).__init__()
		self.num_inputs = num_inputs

	def forward(self, input_images, output_images):
		input_images = [input_images[i] for i in input_images]
		fusion_images = output_images['Fusion']
		tv_loss = 0
		for i in range(self.num_inputs):
			input_image = input_images[i]
			H, W = input_image.shape[2], input_image.shape[3]
			R = input_image - fusion_images
			L_tv = torch.pow(R[:, :, 1:H, :] - R[:, :, 0:H - 1, :], 2).sum() + \
			       torch.pow(R[:, :, :, 1:W] - R[:, :, :, 0:W - 1], 2).sum()
			tv_loss += L_tv
		return tv_loss


if __name__ == '__main__':
	loss = TV_Loss(num_inputs=2)
	vis_images = torch.rand(2, 1, 256, 256)
	inf_images = torch.rand(2, 1, 256, 256)
	fusion_images = torch.rand(2, 1, 256, 256)
	print(loss({'0': vis_images, '1': inf_images}, fusion_images))
