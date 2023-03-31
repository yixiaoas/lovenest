import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader

PATH = "cls10.pt"
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# 添加一层全连接层，将特征矩阵维数转为标签数
model.fc = torch.nn.Linear(num_ftrs, 12)
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)

checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

Transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = datasets.ImageFolder("test", Transform)
test_dataset_loader = DataLoader(test_dataset, batch_size=6, shuffle=True)