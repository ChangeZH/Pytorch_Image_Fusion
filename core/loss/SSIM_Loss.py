import torch
import torch.nn as nn
import torch.nn.functional as F


class SSIM_Loss(nn.Module):
	"""docstring for SSIM_Loss"""

	def __init__(self, sensors, num_channels=3, C=9e-4, device='cuda:0'):
		super(SSIM_Loss, self).__init__()
		self.sensor = sensors
		self.num_channels = num_channels
		self.device = device
		self.c = C

	def forward(self, input_images, output_images):
		batch_size, num_channels = input_images[self.sensor[0]].shape[0], input_images[self.sensor[0]].shape[1]
		ssim_loss = 0
		for sensor in input_images:
			for batch in range(batch_size):
				input_image, output_image = input_images[sensor][batch], output_images[sensor][batch]

				input_image_mean = torch.mean(input_image, dim=[1, 2])
				output_image_mean = torch.mean(output_image, dim=[1, 2])
				C = torch.ones_like(input_image_mean) * self.c

				input_image_var = torch.mean(input_image ** 2, dim=[1, 2]) - input_image_mean ** 2
				input_image_std = input_image_var ** .5

				output_image_var = torch.mean(output_image ** 2, dim=[1, 2]) - output_image_mean ** 2
				output_image_std = output_image_var ** .5

				input_output_var = torch.mean(input_image * output_image,
				                              dim=[1, 2]) - input_image_mean * output_image_mean

				l = (2 * input_image_mean * output_image_mean + C) / (input_image_mean ** 2 + output_image_mean ** 2 + C)
				c = (2 * input_image_std * output_image_std + C) / (input_image_std ** 2 + output_image_std ** 2 + C)
				s = (input_output_var + 2 * C) / (input_image_std * output_image_std + 2 * C)

				ssim_loss += 1 - l * c * s

		return ssim_loss.mean()


if __name__ == '__main__':
	loss = SSIM_Loss(num_channels=3, C=9e-4, device='cpu')
	vis_images = torch.rand(2, 3, 256, 256)
	fusion_images = torch.rand(2, 3, 256, 256)
	print(loss(vis_images, fusion_images))
