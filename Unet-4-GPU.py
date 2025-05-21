import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import json
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import warnings

class UNET(nn.Module):
    def __init__(self, in_channels=3, classes=1):
        super(UNET, self).__init__()
        
        self.layers = [in_channels, 32, 64, 128]
        
        # Define layers without assigning to specific GPUs
        self.encoder1 = self.__double_conv(self.layers[0], self.layers[1]).to('cuda:0')
        self.encoder2 = self.__double_conv(self.layers[1], self.layers[2]).to('cuda:0')
        self.encoder3 = self.__double_conv(self.layers[2], self.layers[3]).to('cuda:1')
        
        self.decoder1 = self.__double_conv(self.layers[3], self.layers[2]).to('cuda:1')
        self.decoder2 = self.__double_conv(self.layers[2], self.layers[1]).to('cuda:1')
        
        self.final_conv = nn.Conv2d(self.layers[1], classes, kernel_size=1).to('cuda:2')

        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2).to('cuda:2')

    def __double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Check input dimensions at each stage
#         print("Input to encoder1:", x.shape)
        x1 = self.encoder1(x)  # On DEVICE
        x2 = self.max_pool(x1)
#         print("Input to encoder2:", x2.shape)
        x2 = self.encoder2(x2)  # On DEVICE

        x3 = self.max_pool(x2)
#         print("Input to encoder3:", x3.shape)
        x3 = self.encoder3(x3)  # On DEVICE

        # Decoder
#         print("Input to decoder1:", x3.shape)
        x4 = self.decoder1(x3)
        x5 = torch.nn.functional.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=True)
#         print("Input to decoder2:", x5.shape)
        x5 = self.decoder2(x5)

        x6 = torch.nn.functional.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=True)
#         print("Input to final_conv:", x6.shape)
        x6 = self.final_conv(x6)

        return x6


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
    model.train()
    for batch in tqdm(data_train): 
        X, y = batch
        X, y = X.to(device), y.to(device)

        # Debugging tensor dimensions
#         print("Train batch X shape:", X.shape, "y shape:", y.shape)
        
        preds = model(X)
#         print("Predictions shape:", preds.shape)

        loss = loss_fn(preds, torch.squeeze(y, 1).type(torch.LongTensor).to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    return loss.item()
        
def main():
    IMAGE_SIZE = 128
    data_transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor()
    ])

    train_data_dir = "/courses/CSYE7105.202510/students/gaali.v/Final-Project/data/train1"
    val_data_dir = "/courses/CSYE7105.202510/students/gaali.v/Final-Project/data/val1"

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

    LEARNING_RATE = 0.0005
    EPOCHS = 5
    global epoch
    epoch = 0
    LOSS_VALS = []
    
    train_set = train_dl
    val_set = val_dl

    print('Data Loaded Successfully!')

    unet = UNET(in_channels=1, classes=3).to(DEVICE)
    optimizer = torch.optim.Adam(unet.parameters(), lr=LEARNING_RATE)
    loss_function = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 10, 100]).to(DEVICE))

    for e in range(epoch, EPOCHS):
        print(f'Epoch: {e+1}/{EPOCHS}')
        loss_train = train_val_function(
            train_set, val_set, unet, 
            optimizer, loss_function, DEVICE
        )
        LOSS_VALS.append([loss_train]) 
        print('Train loss:', loss_train)
        print("Epoch completed and model successfully saved!")
    return unet

if __name__ == "__main__":
    start_time1 = time.time()
    model = main()
    end_time1 = time.time()
    print(f"Total training time: {end_time1 - start_time1} seconds")
