import os
import torch
from tqdm import tqdm
from core.util import debug
from torch.utils.data import DataLoader
import torchvision.transforms as transforms


def test(model, fusion_datasets, configs, load_weight_path=False, save_path=None):
    model.eval()

    if load_weight_path:
        assert configs['TEST']['weight_path'] != 'None', 'Test Need To Resume Chekpoint'
        weight_path = configs['TEST']['weight_path']
        checkpoint = torch.load(weight_path)
        model.load_state_dict(checkpoint['model'].state_dict())
    is_use_gpu = torch.cuda.is_available()

    test_dataloader = DataLoader(fusion_datasets, batch_size=configs['TEST']['batch_size'], shuffle=False)
    test_num_iter = len(test_dataloader)
    dtransforms = transforms.Compose([transforms.ToPILImage()])

    with tqdm(total=test_num_iter) as test_bar:
        for iter, data in enumerate(test_dataloader):

            if is_use_gpu:
                model = model.cuda()
                data = {sensor: data[sensor].cuda() for sensor in data}

            fusion_image = model(data)

            input_imgs, fusion_imgs = debug(configs['MODEL'], configs['TEST_DATASET'], data, fusion_image)
            input_imgs = [input_imgs[sensor] for sensor in configs['MODEL']['input_sensors']]
            imgs = input_imgs + [fusion_imgs]
            imgs = torch.cat(imgs, dim=3)
            for batch in range(imgs.shape[0]):
                if save_path is None:
                    save_path = configs['TEST']['save_path']
                name = os.path.join(save_path, str(len(os.listdir(save_path))))
                img = imgs[batch].cpu()
                img = dtransforms(img)
                img.save(f'{name}.jpg')
            test_bar.update(1)
