import os
from torchvision.io import read_image
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

class myImageDataset(Dataset):
    def __init__(self, img_dir, img_label_dir, transform=None):
        super().__init__()
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(img_label_dir)  # 这是一个dataframe，0是文件名，1是类别
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, index):
        label = self.img_labels.iloc[index, 1]
        #print(label)
        img_path = os.path.join(self.img_dir + f'{label}' + '\\' + self.img_labels.iloc[index, 0])
        image = read_image(img_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def tensorToming(img_tensor):
    img = img_tensor.numpy()
    img = np.transpose(img_tensor, [1, 2, 0])
    plt.imshow(img)

label_dic = {
    0:'Bedroom',
    1:'Diningroom',
    2:'Edwardian Suite',
    3:'Exterior',
    4:'Fitzgerald Suite',
    5:'Hardenbergh Terrace Suite',
    6:'Plaza Suite',
    7:'Regular',
    8:'Royal Suite',
    9:'The Eloise Suite',
    10:'The Palm Court',
    11:'The Plaza Food Hall',
    12:'The Royal Plaza Suite'

}
label_path = '../Hotel1_image\image_data2.csv'
img_root_path = '../Hotel1_image\\'

dataset = myImageDataset(img_root_path, label_path)

# image, label = dataset.__getitem__(33)
# print(image.shape)
# print(label_dic[label])
# tensorToming(image)

#调整图片大小
transform = transforms.Resize((224, 224))

dataset = myImageDataset(img_root_path, label_path, transform)
dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
for imgs, labels in dataloader:
    print(imgs.shape)
    print(labels)
    break

