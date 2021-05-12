import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score


class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 4, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(2),

            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(32 * 6 * 6, 128),
            nn.ReLU(inplace=True),

            nn.Linear(128, 10),
            nn.ReLU(inplace=True)
        )

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

    # all = 0
    # true = 0
    label_list = []
    pred_list = []
    pairwise_distance_list = []

    man_dir = "./data/Reid/Test/sparse_val/"
    sub_dirs = []
    for i in os.listdir(man_dir):
        sub_dirs.append(i + '/')
    # print(len(sub_dirs))

    for sub_dir in sub_dirs:
        files = []

        for file in os.listdir(man_dir + sub_dir):
            files.append(file)

        for i in range(len(files)):
            for j in range(i + 1, len(files)):
                # all += 1
                path1 = man_dir + sub_dir + files[i]
                path2 = man_dir + sub_dir + files[j]
                # print(path1)
                # print(path2)

                label = 0  # 不同
                if path1[-7] == path2[-7]:
                    label = 1  # 相同
                label_list.append(label)

                img1 = torch.from_numpy(np.expand_dims(transform(Image.open(path1).convert("RGB")), 0)).cuda()
                img2 = torch.from_numpy(np.expand_dims(transform(Image.open(path2).convert("RGB")), 0)).cuda()

                output1, output2 = model(img1.cuda(), img2.cuda())
                pairwise_distance = F.pairwise_distance(output1, output2)
                pairwise_distance_list.append(pairwise_distance.data.cpu().numpy()[0])

                # print("ground truth =", label)
                # print("pairwise distance =", pairwise_distance.data.cpu().numpy()[0])
                # print()

    label_list = np.array(label_list)
    pairwise_distance_list = np.array(pairwise_distance_list)
    print("label1 =", np.sum(label_list))
    print("label0 =", len(label_list) - np.sum(label_list))
    print()

    for threshold in np.linspace(0.1, 3.0, 30):
        pred_list = []
        for i in pairwise_distance_list:
            pred = 0
            if i < threshold:
                pred = 1
            pred_list.append(pred)

        pred_list = np.array(pred_list)
        print("threshold =", threshold)
        print("accuracy =", accuracy_score(label_list, pred_list))
        print("precision =", precision_score(label_list, pred_list))
        print("recall =", recall_score(label_list, pred_list))
        print("f1 =", f1_score(label_list, pred_list))
        print("pred1 =", np.sum(pred_list))
        print("pred0 =", len(pred_list) - np.sum(pred_list))
        print()
