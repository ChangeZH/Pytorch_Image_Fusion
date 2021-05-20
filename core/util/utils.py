import yaml
import torch


def load_config(filename):
	with open(filename, 'r') as f:
		config = yaml.safe_load(f)
		return config


def count_parameters(model):
	return sum(p.numel() for p in model.parameters() if p.requires_grad)


def debug(model_config, dataset_config, input_images, fusion_images):
	batch_szie, _, _, _ = input_images[model_config['input_sensors'][0]].shape
	input_imgs = {sensor: [] for sensor in model_config['input_sensors']}
	fusion_imgs = []

	dev = input_images[model_config['input_sensors'][0]].device
	for batch in range(batch_szie):
		img = {sensor: input_images[sensor][batch, :, :, :] for sensor in model_config['input_sensors']}
		fusion = fusion_images['Fusion'][batch, :, :, :]
		channels = fusion.shape[0]
		# std = torch.Tensor(dataset_config['std']).to(dev).view(channels, 1, 1).expand_as(fusion) if channels == 3 \
		#     else torch.Tensor([sum(dataset_config['std']) / len(dataset_config['std'])]).to(dev).view(channels, 1,
		#                                                                                               1).expand_as(
		#     fusion)
		# mean = torch.Tensor(dataset_config['mean']).to(dev).view(channels, 1, 1).expand_as(fusion) if channels == 3 \
		#     else torch.Tensor([sum(dataset_config['mean']) / len(dataset_config['mean'])]).to(dev).view(
		#     channels, 1, 1).expand_as(fusion)
		# img = {sensor: img[sensor] * std + mean for sensor in model_config['input_sensors']}
		# fusion = fusion * std + mean
		img = {sensor: img[sensor] for sensor in model_config['input_sensors']}

		for sensor in model_config['input_sensors']:
			input_imgs[sensor].append(img[sensor])
		fusion_imgs.append(fusion)

	input_imgs = {sensor: torch.stack(input_imgs[sensor], dim=0).to(dev) for sensor in model_config['input_sensors']}
	fusion_imgs = torch.stack(fusion_imgs, dim=0).to(dev)
	return input_imgs, fusion_imgs
