import torch
import torch.nn as nn
from core.model import *


class DenseFuse(nn.Module):
	"""docstring for DenseFuse"""

	def __init__(self, config):
		super(DenseFuse, self).__init__()
		self.config = config
		self.coder = nn.ModuleDict(
			{'Encoder': dense(config['input_channels'], config['out_channels'], config['coder_layers'])})
		self.decoder = nn.ModuleList(
			[nn.Sequential(nn.Conv2d(
				in_channels=min(config['coder_layers'] * config['out_channels'],
				                len(config['input_sensors']) * config['coder_layers'] * config[
					                'out_channels'] // 2 ** i),
				out_channels=len(config['input_sensors']) * config['coder_layers'] * config['out_channels'] // 2 ** (
						i + 1), kernel_size=3, padding=1),
				nn.BatchNorm2d(len(config['input_sensors']) * config['coder_layers'] * config['out_channels'] // 2 ** (
						i + 1)),
				nn.ReLU()) if i != config['decoder_layers'] - 1 else nn.Sequential(
				nn.Conv2d(
					in_channels=len(config['input_sensors']) * config['coder_layers'] * config[
						'out_channels'] // 2 ** i,
					out_channels=config['input_channels'], kernel_size=3, padding=1),
				nn.BatchNorm2d(config['input_channels']),
				nn.ReLU()) for i in range(config['decoder_layers'])])

	def forward(self, inputs, fusion_mode='L1'):
		feats = {}
		for sensor in self.config['input_sensors']:
			feats.update({sensor: self.coder['Encoder'](inputs[sensor])})
		if fusion_mode == 'Add':
			feats = Add_Fusion_Layer(feats)
		elif fusion_mode == 'L1':
			feats = L1_Fusion_Layer(feats)
		for block in self.decoder:
			feats = {sensor: block(feats[sensor]) for sensor in feats}
		return feats
