import os
import cv2
import itertools
import numpy as np


def crop(path_dict, crop_sizes, overlap_sizes, save_path):
    num = 0
    sensors = [i for i in path_dict]
    img_list = {i: os.listdir(path_dict[i]) for i in path_dict}
    for i in sensors:
        if not os.path.exists(os.path.join(save_path, i)):
            os.mkdir(os.path.join(save_path, i))

    for name in img_list[sensors[0]]:
        img = {i: cv2.imread(os.path.join(path_dict[i], name)) for i in sensors}
        img_shape = img[sensors[0]].shape
        for index in range(len(crop_sizes)):
            crop_size = crop_sizes[index]
            overlap_size = overlap_sizes[index]
            y_min = np.arange(0, img_shape[0], crop_size - overlap_size)
            y_min = np.array(list(y_min) + [img_shape[0]])
            x_min = np.arange(0, img_shape[1], crop_size - overlap_size)
            x_min = np.array(list(x_min) + [img_shape[1]])
            y_min = np.unique(np.clip(y_min, a_min=0, a_max=img_shape[0] - crop_size))
            x_min = np.unique(np.clip(x_min, a_min=0, a_max=img_shape[1] - crop_size))
            crop_bboxes = []
            for bbox in itertools.product(y_min, x_min):
                if bbox not in crop_bboxes:
                    crop_bboxes.append(bbox)
                else:
                    continue
            for bbox in crop_bboxes:
                num += 1
                crop_img = {i: img[i][bbox[0]:bbox[0] + crop_size, bbox[1]:bbox[1] + crop_size] for i in img}
                for i in sensors:
                    cv2.imwrite(os.path.join(save_path, i,
                                             name.split('.')[0] + '_' + str(bbox[0]) + '_' + str(bbox[1]) + '.jpg'),
                                crop_img[i])

    print(num)
    return 0


if __name__ == '__main__':
    crop(path_dict={'Vis': '../../datasets/TNO/Vis/', 'Inf': '../../datasets/TNO/Inf/'},
         crop_sizes=[64, 128, 256],
         overlap_sizes=[32, 64, 128],
         save_path='')
