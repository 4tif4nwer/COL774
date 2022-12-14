import os
import sys
import torch
import pandas as pd
from skimage import io
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchvision import  transforms
import time
import copy



class BooksDataset(Dataset):
    
    def __init__(self, x_dataframe, y_dataframe, root_dir, transform=None):
        
        self.book_frame = x_dataframe
        self.book_frame_out = y_dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.book_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(f"{self.root_dir}",self.book_frame.iloc[idx, 1])
        image = io.imread(img_name)
        genre = np.array(self.book_frame_out.iloc[idx, 1])
        sample = {'image': image, 'genre': genre}
        
        if self.transform:
            sample = self.transform(sample)
            

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, genre = sample['image'], sample['genre']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'genre': torch.from_numpy(genre)}

def DataLoad(data_path):
    full_train_x = pd.read_csv(f"{data_path}/train_x.csv")
    full_train_y = pd.read_csv(f"{data_path}/train_y.csv")
    test_x = pd.read_csv(f"{data_path}/non_comp_test_x.csv")
    test_y = pd.read_csv(f"{data_path}/non_comp_test_y.csv")
    train_x,val_x,train_y,val_y = train_test_split(full_train_x,full_train_y,test_size = 0.2,shuffle = True)

    return train_x,train_y,val_x,val_y,test_x,test_y

def CNN_model():
    
    model = nn.Sequential(
    nn.Conv2d(3, 32, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 2),
    nn.Conv2d(32, 64, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 2),
    nn.Conv2d(64, 128, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size = 2),
    nn.Flatten(),
    nn.Linear(128*24*24,128),
    nn.ReLU(),

    nn.Linear(128,30)
    )
    model.to(device)
    return model

def train_model(model, image_datasets, criterion, optimizer, num_epochs=25):

    dataloaders = {x: DataLoader(image_datasets[x], batch_size=64,shuffle = True)
              for x in ['train','val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    since = time.time()

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for samples in dataloaders[phase]:
                inputs = samples['image']
                labels = samples['genre']
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs.float()/255)
                    _, preds = torch.max(outputs, 1)
                    loss = F.cross_entropy(outputs, labels)
                    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        optimizer.zero_grad()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')
    best_model_wts = copy.deepcopy(model.state_dict())
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def main():
    
    global device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    data_path = sys.argv[1]
    train_x,train_y,val_x,val_y,test_x,test_y = DataLoad(data_path)
    train_data = BooksDataset(train_x,train_y,f"{data_path}/images/images",transform=transforms.Compose([ToTensor()]))
    val_data = BooksDataset(val_x,val_y,f"{data_path}/images/images",transform = transforms.Compose([ToTensor()]))
    test_data = BooksDataset(test_x,test_y,f"{data_path}/images/images",transform=transforms.Compose([ToTensor()]))

    image_datasets = {'train' : train_data,'val' : val_data}

    model = CNN_model()

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.Adam(model.parameters(), lr=0.001)

    model_ft = train_model(model,image_datasets, criterion, optimizer_ft,num_epochs=10)   

    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=64,shuffle = False)

    # Save test results to csv

    model_ft.eval()
    test_pred = []
    for samples in test_dataloader:
        inputs = samples['image']
        inputs = inputs.to(device)
        outputs = model_ft(inputs.float()/255)
        _, preds = torch.max(outputs, 1)
        test_pred.extend(preds.cpu().numpy())
    
    test_pred = np.array(test_pred)
    test_y = np.array(test_y['Genre'])
    test_acc = np.sum(test_pred == test_y)/len(test_y)
    print(f'Test Accuracy: {test_acc:.4f}')
    test_df = pd.DataFrame({'id':test_x['Id'],'genre':test_pred})
    test_df.to_csv(f'{data_path}/non comp test pred y',index = False)
            


if __name__ == '__main__':
    main()