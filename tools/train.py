import os
import torch
from tqdm import tqdm
from core.loss import *
from core.util import debug
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


def train(model, train_datasets, test_datasets, configs):
	if not os.path.exists(os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name'])):
		os.mkdir(os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name']))

	model.train()

	train_writer = SummaryWriter(log_dir=os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name']))
	print(f'Run Tensorboard:\n tensorboard --logdir=' + configs['PROJECT']['save_path'] + '/' + configs['PROJECT'][
		'name'] + '/')

	if configs['TRAIN']['resume'] == 'None':
		start_epoch = 1
	else:
		start_epoch = torch.load(configs['TRAIN']['resume'])['epoch'] + 1

	is_use_gpu = torch.cuda.is_available()

	optimizer = eval('torch.optim.' + configs['TRAIN']['opt'])(model.parameters(), configs['TRAIN']['lr'])
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=configs['TRAIN']['milestones'],
	                                                 gamma=configs['TRAIN']['gamma'])

	train_dataloader = DataLoader(train_datasets, batch_size=configs['TRAIN']['batch_size'], shuffle=True)
	train_num_iter = len(train_dataloader)

	loss_func = [eval(l)(sensors=configs['TRAIN_DATASET']['sensors']) for l in configs['TRAIN']['loss_func']]

	all_iter = 0
	for epoch in range(start_epoch, configs['TRAIN']['max_epoch'] + 1):

		loss_epoch = 0

		with tqdm(total=train_num_iter) as train_bar:
			for iter, data in enumerate(train_dataloader):

				if is_use_gpu:
					model = model.cuda(configs['TRAIN']['gpu_id'])
					data = {sensor: data[sensor].cuda(configs['TRAIN']['gpu_id']) for sensor in data}

				fusion_image = model(data)

				loss = [l(data, fusion_image) * configs['TRAIN']['loss_weights'][loss_func.index(l)] for l in loss_func]

				loss_batch = sum(loss)

				loss_epoch += loss_batch.item()
				optimizer.zero_grad()
				loss_batch.backward()
				optimizer.step()

				train_writer.add_scalar('loss', loss_batch, global_step=all_iter)
				train_bar.set_description(
					'Epoch: {}/{}. TRAIN. Iter: {}/{}. All loss: {:.5f}'.format(
						epoch, configs['TRAIN']['max_epoch'], iter + 1, train_num_iter,
						                                      loss_epoch / train_num_iter))
				if configs['TRAIN']['debug_interval'] is not None and all_iter % configs['TRAIN'][
					'debug_interval'] == 0:
					input_imgs, fusion_imgs = debug(configs['MODEL'], configs['TRAIN_DATASET'], data, fusion_image)
					input_imgs = [input_imgs[sensor] for sensor in configs['MODEL']['input_sensors']]
					imgs = input_imgs + [fusion_imgs]
					train_writer.add_image('debug', torch.cat(imgs, dim=2), all_iter, dataformats='NCHW')

				all_iter += 1
				train_bar.update(1)

			scheduler.step()

			train_writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], global_step=epoch)

			if configs['TRAIN']['val_interval'] is not None and all_iter % configs['TRAIN']['val_interval'] == 0:
				torch.save({'model': model, 'epoch': epoch},
				           os.path.join(configs['PROJECT']['save_path'], configs['PROJECT']['name'],
				                        f'model_{epoch}.pth'))
