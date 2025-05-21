import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import os
import numpy as np
from PIL import Image
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import warnings
from tqdm import tqdm

import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF 
CUDA_LAUNCH_BLOCKING=1.

class UNET(nn.Module):
    def __init__(self, in_channels=3, classes=1, base_model=None):
        """
        Initializes the UNET model.

        Args:
            in_channels (int): Number of input channels.
            classes (int): Number of output classes.
            base_model (nn.Module, optional): A pre-trained model to be used as the base.
        """
        super(UNET, self).__init__()
        
        if base_model is None:
            self.layers = [in_channels, 32, 64, 128]
            self.double_conv_downs = nn.ModuleList(
                [self.__double_conv(layer, layer_n) for layer, layer_n in zip(self.layers[:-1], self.layers[1:])])
        else:
            self.layers = []
            self.base_layers = list(base_model.children())
            double_conv_downs_list = []

            # Properly extracting convolutional layers and their out_channels
            # conv1 layer
            conv1 = self.base_layers[0]
            double_conv_downs_list.append(nn.Sequential(conv1))
            self.layers.append(conv1.out_channels)

            # layer1, layer2, layer3, layer4 from ResNet
            for layer in [self.base_layers[4], self.base_layers[5], self.base_layers[6], self.base_layers[7]]:
                double_conv_downs_list.append(layer)
                self.layers.append(layer[-1].bn2.num_features)  # Extracting out_channels from BatchNorm layer after conv2

            self.double_conv_downs = nn.ModuleList(double_conv_downs_list)
        
        self.up_trans = nn.ModuleList(
            [nn.ConvTranspose2d(layer, layer_n, kernel_size=2, stride=2, output_padding=1)
             for layer, layer_n in zip(self.layers[::-1][:-2], self.layers[::-1][1:-1])])
        
        self.double_conv_ups = nn.ModuleList(
            [self.__double_conv(layer, layer // 2) for layer in self.layers[::-1][:-2]])
        
        # Updated pooling to include padding to prevent size reduction to zero
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        
        self.final_conv = nn.Conv2d(self.layers[1], classes, kernel_size=1)

    def __double_conv(self, in_channels, out_channels):
        """
        Creates a double convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        
        Returns:
            nn.Sequential: A sequential model with two convolutional layers.
        """
        conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return conv
    
    def forward(self, x):
        """
        Forward pass through the UNET model.

        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Output tensor.
        """
        concat_layers = []
        
        for down in self.double_conv_downs:
            x = down(x)
            if down != self.double_conv_downs[-1]:
                concat_layers.append(x)
                # Ensure pooling does not reduce spatial dimensions to zero
                if x.shape[2] > 1 and x.shape[3] > 1:
                    x = self.max_pool_2x2(x)
        
        concat_layers = concat_layers[::-1]
        for up_trans, double_conv_up, concat_layer in zip(self.up_trans, self.double_conv_ups, concat_layers):
            x = up_trans(x)
            if x.shape != concat_layer.shape:
                x = TF.resize(x, concat_layer.shape[2:], interpolation=TF.InterpolationMode.BILINEAR)
            
            concatenated = torch.cat((concat_layer, x), dim=1)
            x = double_conv_up(concatenated)
        
        x = self.final_conv(x)
        
        return x

class DatasetLiverCT(Dataset):
  def __init__(self, data_dir, mode='train', transform = None, 
               liver_thr = 40, tumor_thr = 230) -> None:
    self.data_dir = data_dir
    self.data_path = data_dir + '/data/'
    self.labels_path = data_dir + '/labels/'
    self.mode = mode
    self.transform = transform
    self.tumor_thr = tumor_thr
    self.liver_thr = liver_thr
  
  def __len__(self):
    return len(os.listdir(self.data_dir + '/data'))
  
  def __getitem__(self, idx):
    img_path = self.data_path + os.listdir(self.data_path)[idx]
    img = Image.open(img_path)
    if self.transform is not None:
      img = self.transform(img)
    img = np.array(img)
    if self.mode == 'train':
      label_path = self.labels_path + img_path.split('/')[-1]
      label = Image.open(label_path).convert("L")
      if self.transform is not None:
        label = self.transform(label)

      label = np.array(label) * 255
      label[label < self.liver_thr] = 0
      label[(label >= self.liver_thr) & (label < self.tumor_thr)] = 1
      label[label >= self.tumor_thr] = 2

      return img.astype('float32'), label
    else:
      return img.astype('float32')
  
  def get_file_name(self, idx):
    return os.listdir(self.data_path)[idx].split('/')[0]



def train_val_function(data_train, data_val, model, optimizer, loss_fn, device):
    print('Entering into train/val function')
    loss_values = []
    model.train()
    data = tqdm(data_train)
    for index, batch in enumerate(data): 
        X, y = batch
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = loss_fn(
            preds, 
            torch.squeeze(y, 1).type(torch.LongTensor).to(device)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    model.eval()
    data = tqdm(data_val)
    with torch.no_grad():
      for index, batch in enumerate(data): 
          X, y = batch
          X, y = X.to(device), y.to(device)
          preds = model(X)
          loss_val = loss_fn(
              preds, 
              torch.squeeze(y, 1).type(torch.LongTensor).to(device)
          )

    return loss.item(), loss_val.item()
        

def main():
    import torch 
    import torch.optim as optim
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from tqdm import tqdm
    import pdb
    
    IMAGE_SIZE = 128
    data_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    train_data_dir = "/courses/CSYE7105.202510/students/krishnamurthy.p/Final-Project/data/train1"
    val_data_dir = "/courses/CSYE7105.202510/students/krishnamurthy.p/Final-Project/data/val1"

    train_ds = DatasetLiverCT(train_data_dir, transform=data_transform)
    val_ds = DatasetLiverCT(val_data_dir, transform=data_transform)

    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=True, num_workers=0)


    

    if torch.cuda.is_available():
        DEVICE = 'cuda:0'
        print('Running on the GPU')
    else:
        DEVICE = "cpu"
        print('Running on the CPU')

    MODEL_PATH = 'YOUR-MODEL-PATH'
    LOAD_MODEL = False
    ROOT_DIR = '../datasets/cityscapes'
    IMG_HEIGHT = 110  
    IMG_WIDTH = 220  
    BATCH_SIZE = 16 
    LEARNING_RATE = 0.0005
    EPOCHS = 5
    global epoch
    epoch = 0
    LOSS_VALS = []
    
    train_set = train_dl
    val_set = val_dl

    print('Data Loaded Successfully!')

    unet = UNET(in_channels=1, classes=3).to(DEVICE).train()
    optimizer = optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 10, 100]).to(DEVICE))
    for e in range(epoch, EPOCHS):
        print(f'Epoch: {e}')
        loss_train, loss_val = train_val_function(
            train_set, val_set, unet, 
            optimizer, loss_function, DEVICE
        )
        LOSS_VALS.append([loss_train, loss_val]) 
        print('Train loss:', loss_train, ', Val loss:', loss_val)
        print("Epoch completed and model successfully saved!")
    return unet


import time

start_time1 = time.time()
model = main()
end_time1 = time.time()
print(end_time1-start_time1)