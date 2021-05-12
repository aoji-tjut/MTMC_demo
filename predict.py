import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2 as cv


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 6, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(6),

            nn.ReflectionPad2d(1),
            nn.Conv2d(6, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 100 * 100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss = torch.mean((1 - label) * torch.pow(distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))

        return loss


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNet().to(device)
    model.load_state_dict(torch.load("./model/model.pt"))
    model.eval()
    transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

    man_dir = "./data/Reid/Test/sparse_val/"
    sub_dir = []
    for i in os.listdir(man_dir):
        sub_dir.append(i + '/')

    while True:
        rand_sub_dir = random.choice(sub_dir)
        file = []
        for i in os.listdir(man_dir + rand_sub_dir):
            file.append(i)
        path1 = man_dir + rand_sub_dir + random.choice(file)
        path2 = man_dir + rand_sub_dir + random.choice(file)
        if path1 == path2:
            continue
        print(path1)
        print(path2)

        label = 1  # 不同
        if path1[-7] == path2[-7]:
            label = 0  # 相同
        img1 = torch.from_numpy(np.expand_dims(transform(Image.open(path1).convert("RGB")), 0)).cuda()
        img2 = torch.from_numpy(np.expand_dims(transform(Image.open(path2).convert("RGB")), 0)).cuda()

        car1 = img1.data.cpu().numpy()[0][0]
        car2 = img2.data.cpu().numpy()[0][0]
        cv.imshow("car", np.hstack([car1, np.zeros((100, 20)) + 255, car2]))

        output1, output2 = model(img1.cuda(), img2.cuda())
        pairwise_distance = F.pairwise_distance(output1, output2)
        criterion = ContrastiveLoss()
        loss = criterion(output1, output2, label)

        print("ground truth =", label)
        print("pairwise distance =", pairwise_distance.data.cpu().numpy()[0])
        print("loss =", loss.data.cpu().numpy())
        print()

        cv.waitKey(0)
