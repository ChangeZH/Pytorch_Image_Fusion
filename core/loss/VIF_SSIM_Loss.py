import torch
import torch.nn as nn
import torch.nn.functional as F


class VIF_SSIM_Loss(nn.Module):
	"""docstring for VIF_SSIM_Loss"""

	def __init__(self, sensors, kernal_size=11, num_channels=1, C=9e-4, device='cuda:0'):
		super(VIF_SSIM_Loss, self).__init__()
		self.sensors = sensors
		self.kernal_size = kernal_size
		self.num_channels = num_channels
		self.device = device
		self.c = C

		self.avg_kernal = torch.ones(num_channels, 1, self.kernal_size, self.kernal_size) / (self.kernal_size) ** 2
		self.avg_kernal = self.avg_kernal.to(device)

	def forward(self, input_images, output_images):
		vis_images, inf_images, fusion_images = input_images[self.sensors[0]], input_images[self.sensors[1]], \
		                                        output_images['Fusion']
		batch_size, num_channels = vis_images.shape[0], vis_images.shape[1]

		vis_images_mean = F.conv2d(vis_images, self.avg_kernal, stride=self.kernal_size, groups=num_channels)
		vis_images_var = torch.abs(F.conv2d(vis_images ** 2, self.avg_kernal, stride=self.kernal_size,
		                                    groups=num_channels) - vis_images_mean ** 2)

		inf_images_mean = F.conv2d(inf_images, self.avg_kernal, stride=self.kernal_size, groups=num_channels)
		inf_images_var = torch.abs(F.conv2d(inf_images ** 2, self.avg_kernal, stride=self.kernal_size,
		                                    groups=num_channels) - inf_images_mean ** 2)

		fusion_images_mean = F.conv2d(fusion_images, self.avg_kernal, stride=self.kernal_size, groups=num_channels)
		fusion_images_var = torch.abs(F.conv2d(fusion_images ** 2, self.avg_kernal, stride=self.kernal_size,
		                                       groups=num_channels) - fusion_images_mean ** 2)

		vis_fusion_images_var = F.conv2d(vis_images * fusion_images, self.avg_kernal, stride=self.kernal_size,
		                                 groups=num_channels) - vis_images_mean * fusion_images_mean
		inf_fusion_images_var = F.conv2d(inf_images * fusion_images, self.avg_kernal, stride=self.kernal_size,
		                                 groups=num_channels) - inf_images_mean * fusion_images_mean

		C = torch.ones_like(fusion_images_mean) * self.c

		ssim_l_vis_fusion = (2 * vis_images_mean * fusion_images_mean + C) / \
		                    (vis_images_mean ** 2 + fusion_images_mean ** 2 + C)
		ssim_l_inf_fusion = (2 * inf_images_mean * fusion_images_mean + C) / \
		                    (inf_images_mean ** 2 + fusion_images_mean ** 2 + C)

		ssim_s_vis_fusion = (vis_fusion_images_var + C) / (vis_images_var + fusion_images_var + C)
		ssim_s_inf_fusion = (inf_fusion_images_var + C) / (inf_images_var + fusion_images_var + C)

		score_vis_inf_fusion = (vis_images_mean > inf_images_mean) * ssim_l_vis_fusion * ssim_s_vis_fusion + \
		                       (vis_images_mean <= inf_images_mean) * ssim_l_inf_fusion * ssim_s_inf_fusion

		ssim_loss = 1 - score_vis_inf_fusion.mean()

		return ssim_loss


if __name__ == '__main__':
	loss = VIF_SSIM_Loss(kernal_size=8, num_channels=1, C=9e-4, device='cpu')
	vis_images = torch.rand(2, 1, 256, 256)
	inf_images = torch.rand(2, 1, 256, 256)
	fusion_images = torch.rand(2, 1, 256, 256)
	print(loss({'Vis': vis_images, 'Inf': inf_images}, fusion_images))
