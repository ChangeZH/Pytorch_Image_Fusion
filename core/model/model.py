import torch
import torch.nn as nn


class dense(nn.Module):
	"""docstring for dense"""

	def __init__(self, in_channels, out_channels, num_layers):
		super(dense, self).__init__()
		self.dense_block = nn.ModuleList([nn.Sequential(
			nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()) if i == 0 else nn.Sequential(
			nn.Conv2d(in_channels=out_channels * i, out_channels=out_channels, kernel_size=3, padding=1),
			nn.BatchNorm2d(out_channels),
			nn.ReLU()) for i in range(num_layers)])

	def forward(self, inputs):
		feats = [inputs]
		for block in self.dense_block:
			feat = block(torch.cat(feats, dim=1)) if len(feats) == 1 else block(torch.cat(feats[1:], dim=1))
			feats.append(feat)
		return torch.cat(feats[1:], dim=1)


def Add_Fusion_Layer(inputs):
	inputs.update({'Fusion': torch.stack([inputs[sensor] for sensor in inputs], dim=0).sum(0)})
	return inputs


def L1_Fusion_Layer(inputs, kernal_size=3):
	avgpool = torch.nn.AvgPool2d(kernal_size, 1, (kernal_size - 1) // 2)
	weights = {sensor: avgpool(inputs[sensor]) for sensor in inputs}
	weights_sum = torch.stack([weights[sensor] for sensor in weights], dim=0).sum(0)
	weights = {sensor: (weights[sensor] + torch.ones_like(weights_sum) * 9e-4) /
	                   (weights_sum + torch.ones_like(weights_sum) * 9e-4) for sensor in inputs}
	inputs.update({'Fusion': torch.stack([weights[sensor] * inputs[sensor] for sensor in inputs], dim=0).sum(0)})
	return inputs
