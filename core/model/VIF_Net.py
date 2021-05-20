import torch
import torch.nn as nn
from core.model import dense


class VIF_Net(nn.Module):
	"""docstring for VIF_Net"""

	def __init__(self, config):
		super(VIF_Net, self).__init__()
		self.config = config
		self.coder = nn.ModuleDict(
			{sensor: dense(config['input_channels'], config['out_channels'], config['coder_layers']) for sensor in
			 config['input_sensors']})
		self.decoder = nn.ModuleList(
			[nn.Sequential(nn.Conv2d(
				in_channels=len(config['input_sensors']) * config['coder_layers'] * config['out_channels'] // 2 ** i,
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

	def forward(self, inputs):
		feats = {}
		for sensor in self.config['input_sensors']:
			feats.update({sensor: self.coder[sensor](inputs[sensor])})
		feats = torch.cat([feats[sensor] for sensor in self.config['input_sensors']], dim=1)
		for block in self.decoder:
			feats = block(feats)
		outputs = {'Fusion': feats}
		for sensor in inputs:
			outputs.update({sensor: inputs[sensor]})
		return outputs
