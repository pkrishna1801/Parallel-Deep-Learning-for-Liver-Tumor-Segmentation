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
CUDA_LAUNCH_BLOCKING = 1 

warnings.filterwarnings("ignore", category=UserWarning)
# UNET model definition
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

# Dataset class
class DatasetLiverCT(Dataset):
    def __init__(self, data_dir, mode='train', transform=None, liver_thr=40, tumor_thr=230):
        """
        Initializes the Liver CT dataset.

        Args:
            data_dir (str): Path to the data directory.
            mode (str): Mode of the dataset ('train' or 'test').
            transform (callable, optional): Transformations to be applied to the images.
            liver_thr (int): Threshold for liver segmentation.
            tumor_thr (int): Threshold for tumor segmentation.
        """
        self.data_dir = data_dir
        self.data_path = os.path.join(data_dir, 'data')
        self.labels_path = os.path.join(data_dir, 'labels')
        self.mode = mode
        self.transform = transform
        self.tumor_thr = tumor_thr
        self.liver_thr = liver_thr

    def __len__(self):
        """
        Returns the length of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return len(os.listdir(self.data_path))

    def __getitem__(self, idx):
        """
        Gets the image (and label if in training mode) for the given index.

        Args:
            idx (int): Index of the sample.
        
        Returns:
            tuple: (image, label) if in training mode, otherwise just image.
        """
        img_path = os.path.join(self.data_path, os.listdir(self.data_path)[idx])
        img = Image.open(img_path)
        if self.transform:
            img = self.transform(img)
        img = np.array(img).astype('float32')
        if self.mode == 'train':
            label_path = os.path.join(self.labels_path, os.path.basename(img_path))
            label = Image.open(label_path).convert("L")
            if self.transform:
                label = self.transform(label)
            label = np.array(label) * 255
            label[label < self.liver_thr] = 0
            label[(label >= self.liver_thr) & (label < self.tumor_thr)] = 1
            label[label >= self.tumor_thr] = 2
            return img, label
        return img

    def get_file_name(self, idx):
        """
        Gets the file name for the given index.

        Args:
            idx (int): Index of the sample.
        
        Returns:
            str: File name of the sample.
        """
        return os.path.basename(os.listdir(self.data_path)[idx])

# Helper functions
def setup(rank, world_size):
    """
    Sets up the environment for distributed training.

    Args:
        rank (int): The rank of the current process.
        world_size (int): The total number of processes involved in the training.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12359"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)  # Set to GPU 1, 2, 3, or 4

def cleanup():
    """
    Cleans up the distributed training environment.
    """
    dist.destroy_process_group()

def get_dataloader(dataset, batch_size, world_size, rank, num_workers=4):
    """
    Creates a DataLoader for distributed training.

    Args:
        dataset (Dataset): Dataset to load.
        batch_size (int): Batch size.
        world_size (int): Number of processes involved in training.
        rank (int): Rank of the current process.
        num_workers (int, optional): Number of worker threads.
    
    Returns:
        DataLoader: DataLoader for the dataset.
    """
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, pin_memory=True, num_workers=num_workers)

# Training function
def train_epoch(data_loader, model, optimizer, loss_fn, device):
    """
    Trains the model for one epoch.

    Args:
        data_loader (DataLoader): DataLoader for training data.
        model (nn.Module): Model to be trained.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        loss_fn (nn.Module): Loss function.
        device (str): Device to run the training on.
    
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0.0
    for X, y in tqdm(data_loader, desc="Training", leave=False):
        X, y = X.to(device), y.to(device)
        preds = model(X)
        loss = loss_fn(preds, torch.squeeze(y, 1).long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

# Validation function
def validate_epoch(data_loader, model, loss_fn, device, rank, epoch, last_epoch=False):
    """
    Validates the model for one epoch.

    Args:
        data_loader (DataLoader): DataLoader for validation data.
        model (nn.Module): Model to be validated.
        loss_fn (nn.Module): Loss function.
        device (str): Device to run the validation on.
        rank (int): Rank of the current process.
        epoch (int): Current epoch number.
        last_epoch (bool, optional): Whether this is the last epoch of training.

    Returns:
        tuple: (average validation loss, average accuracy, metrics for each batch)
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    metrics = []
    images_saved = 0
    
    try:
        with torch.no_grad():
            for idx, (X, y) in enumerate(tqdm(data_loader, desc="Validation", leave=False)):
                X, y = X.to(device), y.to(device)
                preds = model(X)
                loss = loss_fn(preds, torch.squeeze(y, 1).long())
                total_loss += loss.item()
                
                # Calculate accuracy
                pred_classes = preds.argmax(dim=1).cpu().numpy()
                true_classes = y.cpu().numpy()
                accuracy = accuracy_score(true_classes.flatten(), pred_classes.flatten())
                total_accuracy += accuracy
                
                # Save input masks and predictions for all ranks (5-6 images) in the last epoch only
                if last_epoch and images_saved < 6:  # Save only the first 5-6 images per rank
                    os.makedirs('saved_results', exist_ok=True)
                    input_image = X.cpu().numpy()
                    mask = y.cpu().numpy()
                    output_pred = pred_classes
                    
                    # Save for each rank
                    gpu_num = rank
                    np.save(f'saved_results/input_image_gpu{gpu_num}_batch_{idx}.npy', input_image)
                    np.save(f'saved_results/mask_gpu{gpu_num}_batch_{idx}.npy',mask)
                    np.save(f'saved_results/output_gpu{gpu_num}_batch_{idx}.npy', output_pred)
                    images_saved += 1
                
                metrics.append({
                    'input_idx': idx,
                    'loss': loss.item(),
                    'accuracy': accuracy
                })
                
            if len(data_loader) > 0:
                avg_accuracy = total_accuracy / len(data_loader)
            else:
                avg_accuracy = 0.0
                
    except Exception as e:
        print(f"An error occurred during validation on rank {rank}: {e}")
        avg_accuracy = 0.0
        
    return total_loss / len(data_loader), avg_accuracy, metrics

# Main training loop

def train_model(rank, world_size, train_data_dir, val_data_dir, epochs=4, batch_size=32, learning_rate=0.0005, all_metrics=None):
    """
    Trains the model across multiple epochs and multiple processes (GPUs).

    Args:
        rank (int): Rank of the current process.
        world_size (int): Number of processes involved in training.
        train_data_dir (str): Path to training data directory.
        val_data_dir (str): Path to validation data directory.
        epochs (int, optional): Number of epochs to train for.
        batch_size (int, optional): Batch size for training.
        learning_rate (float, optional): Learning rate for optimizer.
        all_metrics (list, optional): List to store metrics for all processes.
    """
    try:
        setup(rank, world_size)
        device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"
        print(f"Running GPU {rank if torch.cuda.is_available() else 'CPU'} (rank {rank })")

        transform = transforms.Compose([
            transforms.Resize(128),
            transforms.ToTensor()
        ])

        train_ds = DatasetLiverCT(train_data_dir, transform=transform)
        val_ds = DatasetLiverCT(val_data_dir, transform=transform)
        train_dl = get_dataloader(train_ds, batch_size, world_size, rank)
        val_dl = get_dataloader(val_ds, batch_size, world_size, rank)
        
        torch.backends.cudnn.enabled = False
        model = UNET(in_channels=1, classes=3).to(device)
        model = model.to(rank)
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss(weight=torch.tensor([0.5, 10, 100]).to(device))

        start_time = time.time()
        gpu_metrics = {'rank': rank, 'epochs': []}
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}/{epochs}')
            train_loss = train_epoch(train_dl, model, optimizer, loss_function, device)
            last_epoch = (epoch == epochs - 1)
            val_loss, val_accuracy, epoch_metrics = validate_epoch(val_dl, model, loss_function, device, rank, epoch, last_epoch=last_epoch)
            print(f'Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')
            
            # Collect metrics for the entire training
            epoch_info = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'metrics': epoch_metrics
            }
            gpu_metrics['epochs'].append(epoch_info)
            
        end_time = time.time()
        
        total_duration = end_time - start_time
        gpu_metrics['total_duration'] = total_duration
        if all_metrics is not None:
            all_metrics.append(gpu_metrics)
            
        # Save metrics to JSON file
        os.makedirs('results', exist_ok=True)
        with open(f'results/metrics_gpu{rank }.json', 'w') as f:
            json.dump(gpu_metrics, f)
            
    except Exception as e:
        print(f"An error occurred on rank {rank}: {e}")
    finally:
        cleanup()




# Main function to manage training across multiple processes
def main():
    """
    Main function to initiate training across multiple GPUs.
    """
    gpus = [1, 2, 3, 4]  # Use GPUs 1, 2, 3, and 4
    train_data_dir = "/courses/CSYE7105.202510/students/krishnamurthy.p/Final-Project/data/train1"
    val_data_dir = "/courses/CSYE7105.202510/students/krishnamurthy.p/Final-Project/data/val1"
    all_metrics = []
    epochs = 20
    batch_size = 32
    learning_rate = 0.005
    excuted_times = []
   

    for gpu in gpus:
        start_time1 = time.time()
        print(f"Currently running on {gpu} GPUs ")
        mp.spawn(train_model, args=(gpu, train_data_dir, val_data_dir, epochs, batch_size, learning_rate, all_metrics), nprocs=gpu, join=True)
        print(f"Completed training with {gpu} GPU(s)\n")
        end_time1 = time.time()
        excuted_times.append(end_time1-start_time1)
        print(end_time1-start_time1)
    
    # Save metrics to JSON file
    os.makedirs('results', exist_ok=True)
    with open(f'metrics_gpu.json', 'w') as f:
        json.dump(excuted_times, f)

if __name__ == "__main__":
    main()