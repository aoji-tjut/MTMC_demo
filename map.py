import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import functools


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


def comp(a, b):
    if a[0] < b[0]:
        return True
    else:
        return False


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNet().to(device)
    model.load_state_dict(torch.load("./model/model.pt"))
    model.eval()
    transform = transforms.Compose([transforms.Resize((100, 100)), transforms.ToTensor()])

    # label_list = []
    # pred_list = []
    # pairwise_distance_list = []
    map_list = []

    man_dir = "./data/Reid/Test/sparse_val/"
    sub_dirs = []
    for i in os.listdir(man_dir):
        sub_dirs.append(i + '/')
    print(len(sub_dirs))

    for sub_dir in sub_dirs:
        files = []

        for file in os.listdir(man_dir + sub_dir):
            files.append(file)

        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                path1 = man_dir + sub_dir + files[i]
                path2 = man_dir + sub_dir + files[j]
                print(path1)
                print(path2)

                label = 1  # 不同
                if path1[-7] == path2[-7]:
                    label = 0  # 相同
                # label_list.append(label)

                img1 = torch.from_numpy(np.expand_dims(transform(Image.open(path1).convert("RGB")), 0)).cuda()
                img2 = torch.from_numpy(np.expand_dims(transform(Image.open(path2).convert("RGB")), 0)).cuda()

                output1, output2 = model(img1.cuda(), img2.cuda())
                pairwise_distance = F.pairwise_distance(output1, output2)
                # pairwise_distance_list.append(pairwise_distance)

                print("ground truth =", label)
                print("pairwise distance =", pairwise_distance.data.cpu().numpy()[0])
                print()

                pred = 1
                if pairwise_distance < 1.0:
                    pred = 0
                # pred_list.append(pred)

                if label == pred:
                    map_list.append([pairwise_distance.data.cpu().numpy()[0], 1])
                else:
                    map_list.append([pairwise_distance.data.cpu().numpy()[0], 0])

    map_list = sorted(map_list, key=lambda x: (x[0]))
    map_list = np.array(map_list)
    true = 1
    sum = 0
    for num in range(len(map_list)):
        if map_list[num][1]:
            sum += true / (num + 1)
            true += 1

    print(sum / len(map_list))
