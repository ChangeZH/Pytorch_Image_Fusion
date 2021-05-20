# Pytorch_Image_Fusion  
&emsp;&emsp;基于Pytorch框架的多源图像像素级融合，包含针对多种网络的复现。  
&emsp;&emsp;The pixel level fusion of multi-source images based on the pytorch framework includes the reproduction of multiple networks.  
&emsp;&emsp;  详细请访问 👉 https://blog.csdn.net/qq_36449741/article/details/104406931  
  
![results](https://github.com/ChangeZH/Pytorch_Image_Fusion/blob/main/img/results.jpg)

## 环境要求 / Environmental Requirements  
  
```
conda create -n PIF python=3.7  
conda activate PIF  
conda install pytorch=1.6.0 torchvision -c pytorch  
pip install tqdm pyyaml tensorboardX opencv-python  
```
  
## 数据集 / Dataset  
  ⚡ TNO数据集下载地址 👉  链接：https://pan.baidu.com/s/1-6b-0onDCEPHAMUWyEkmtA  提取码：`PIF0`   

  注意要保证在不同数据类型文件夹下图片名称完全一样。
  提供切片裁剪程序  ` ./core/dataset/crop_datasets.py `  ，修改：
  ```python
  # 此文件为./core/dataset/crop_datasets.py 45行
  
  if __name__ == '__main__':
    crop(path_dict={'Vis': '../../datasets/TNO/Vis/', 'Inf': '../../datasets/TNO/Inf/'}, # 数据类型与其路径的对应字典，例如'Vis'数据的路径为'../../datasets/TNO/Vis/'，'Inf'数据的路径为'../../datasets/TNO/Inf/'，类型为字典
         crop_sizes=[64, 128, 256], # 切片大小，可以多种大小，类型为列表
         overlap_sizes=[32, 64, 128], # 切片重叠区域大小，与切片大小对应，不能大于对应切片大小，类型为列表
         save_path='') # 切片保存位置，类型为字符串
  ```  
  修改后运行  ` python crop_datasets.py `  进行数据切片。
  
## 参数设置 / Parameter Setting  
  
```python
# 此文件为./config/VIF_Net.yaml

PROJECT: # 项目参数
  name: 'VIF_Net_Image_Fusion' # 项目名称
  save_path: './work_dirs/' # 项目保存路径，训练模型会保存至此路径下的项目名称文件夹中

TRAIN_DATASET: # 训练数据集参数
  root_dir: './datasets/TNO_crop/' # 训练数据集根目录
  sensors: [ 'Vis', 'Inf' ] # 训练数据集包含的数据类型
  channels: 1 # 训练数据中图片的通道数
  input_size: 128 # 训练数据中图片的尺寸
  mean: [ 0.485, 0.456, 0.406 ] # 训练数据中图片的归一化均值（暂时用不到）
  std: [ 0.229, 0.224, 0.225 ] # 训练数据中图片的归一化标准差（暂时用不到）

TRAIN: # 训练参数
  batch_size: 32 # 训练批次大小
  max_epoch: 200 # 训练最大代数
  lr: 0.01 # 训练学习率
  gamma: 0.01 # 训练学习率衰减系数
  milestones: [ 100, 150, 175 ] # 训练学习率衰减的里程碑
  opt: Adam # 训练优化器
  loss_func: ['VIF_SSIM_Loss', 'TV_Loss'] # 训练使用的损失函数
  val_interval: 1 # 训练每过多少代数后保存权重
  debug_interval: 100 # 训练每过多少批次后进行可视化，结果可视化在tensorboard中
  resume: None # 训练停止后继续训练加载权重路径
  loss_weights: [ 1000, 1 ] # 对VIF_Net的两个损失的权值

TEST_DATASET: # 测试数据集参数
  root_dir: './datasets/TNO/' # 测试数据集根目录
  sensors: [ 'Vis', 'Inf' ] # 测试数据集包含的数据类型
  channels: 1 # 测试数据中图片的通道数
  input_size: 512 # 测试数据中图片的尺寸
  mean: [ 0.485, 0.456, 0.406 ] # 测试数据中图片的归一化均值（暂时用不到）
  std: [ 0.229, 0.224, 0.225 ] # 测试数据中图片的归一化标准差（暂时用不到）

TEST: # 测试参数
  batch_size: 2 # 测试批次大小
  weight_path: './work_dirs/VIF_Net_Image_Fusion/model_50.pth' # 测试加载的权重路径
  save_path: './test/' # 测试结果保存路径

MODEL: # 模型参数
  model_name: 'VIF_Net' # 模型名称
  input_channels: 1 # 模型输入通道数
  out_channels: 16 # 模型每一层输出的通道数
  input_sensors: [ 'Vis', 'Inf' ] # 模型输入数据类型
  coder_layers: 4 # 模型编码器层数
  decoder_layers: 4 # 模型解码器层数

```  

## 训练与测试 / Training And Testing  
  
### 训练 / Training  
&emsp;&emsp;运行  ` python run.py --train `  进行训练。训练的模型权重会保存再指定的路径下。  

#### tensorboardX进行训练可视化  
&emsp;&emsp;运行  ` tensorboard --logdir= XXX `  进行训练可视化。将  ` XXX `  替换为模型储存的路径。例如，config中有如下参数：  
```python
PROJECT:
  name: 'VIF_Net_Image_Fusion'
  save_path: './work_dirs/'
  weight_path: ''
```  
&emsp;&emsp;可运行  ` tensorboard --logdir= ./work_dirs/VIF_Net_Image_Fusion/ `  进行训练可视化。再次训练后最好删除之前的  ` events `  文件。  
![SCALARS](https://github.com/ChangeZH/Pytorch_Image_Fusion/blob/main/img/TensorBoard_0.png)
![IMAGES](https://github.com/ChangeZH/Pytorch_Image_Fusion/blob/main/img/TensorBoard_1.png)  
&emsp;&emsp;上图中每三行为一组，前两行为输入数据，第三行为融合结果。  
  
### 测试 / Testing  
&emsp;&emsp;运行  ` python run.py --test `  进行测试。结果会批量保存至指定路径下。  

## 预训练模型 / Pre-training Model
 - [x] ⚡ VIF_Net 👉   链接：https://pan.baidu.com/s/1avjiuNTovsoFmUWd5aPpzg 提取码：PIF2  
 - [ ] ⚡ DenseFuse 👉   
 
## 计划中 / To Do  
 - [x] VIF_Net 👉 https://blog.csdn.net/qq_36449741/article/details/104562999  
 - [ ] DenseFuse 👉 https://blog.csdn.net/qq_36449741/article/details/104776319  
