# -*- coding: UTF-8 -*-
import torch
import torchvision
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import PIL.ImageOps
from torchvision import models
import os
from pathlib import Path
import json
import vis

# test_dir = "../data/pic/test(add)"
IMG_SIZE = 100


# car_frame = []

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


def addlist(reslist, frame, object, bbox, camera_id, color):
    for i in reslist:
        if (i['frame_id'] == int(frame)):
            i['object'] = i['object'] + [
                {'obj_id': int(object), 'camera_id': int(camera_id), 'bbox': bbox, 'color': color}]


class CarList(object):
    def __init__(self):
        self.obj_id = {}
        self.id = []

    def add_obj_id(self, id, img):
        for obj in self.obj_id:
            if (obj == id):
                self.obj_id[id].append(img)
                return
        self.id.append(id)
        self.obj_id[id] = []
        self.obj_id[id].append(img)

    def return_id(self):
        for id in self.id:
            print(id)

    def return_img(self, id):
        return self.obj_id[id]


def addlist(reslist, frame, object, bbox, camera_id, color):
    for i in reslist:
        if (i['frame_id'] == int(frame)):
            i['object'] = i['object'] + [
                {'obj_id': int(object), 'camera_id': int(camera_id), 'bbox': bbox, 'color': color}]


def readframe(frame_id):
    filedir1 = test_dir + "/" + "%06d" % (frame_id) + "/"
    filedir2 = test_dir + "/" + "%06d" % (frame_id) + "/"
    filedir3 = test_dir + "/" + "%06d" % (frame_id) + "/"
    filedir4 = test_dir + "/" + "%06d" % (frame_id) + "/"
    file1 = []
    file2 = []
    file3 = []
    file4 = []
    file = []

    # for root, dirs, files in os.walk(filedir1):
    #     for file in files:
    #         print(file)

    path1 = Path(r"" + filedir1)
    for fe in path1.rglob("*.jpg"):
        fe = str(fe)
        file1.append(fe)

    file.append(file1)  # print(file)
    # print("2&3")
    path2 = Path(r"" + filedir2)
    for fe in path2.rglob("*.jpg"):
        fe = str(fe)
        file2.append(fe)
    file.append(file2)  # print(file)
    # print("3&4")
    path3 = Path(r"" + filedir3)
    for fe in path3.rglob("*.jpg"):
        fe = str(fe)
        file3.append(fe)
    file.append(file3)  # print(file)

    path4 = Path(r"" + filedir4)
    for fe in path4.rglob("*.jpg"):
        fe = str(fe)
        file4.append(fe)
    file.append(file4)

    return file


reslist = []


def main(model_dir, thresold):
    car_list = CarList()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SiameseNet().to(device)
    model.load_state_dict(torch.load(model_dir))
    model.eval()
    transform = transforms.Compose([transforms.Resize((100, 100)),  # 有坑，传入int和tuple有区别
                                    transforms.ToTensor()])
    # print(readframe(0)[0])
    # print(readframe(0)[1])
    # print(readframe(0)[2])
    # print(readframe(0)[3])
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    maxdis = 0
    mindis = 10
    # thresold = 1
    for i in range(100):
        element = {'frame_id': i, 'object': []}
        reslist.append(element)
    for index in range(1200):
        files = readframe(index)
        fileindex = 0
        for file in files:
            fileindex += 1
            if (fileindex == 4):
                for i in range(len(file)):
                    img0name = file[i].split('/')[-1]
                    img0_camera_id = img0name.split('_')[1]
                    # img0_type = img0name.split('_')[8]
                    img0_obj_id = img0name.split('_')[1]
                    ifexist = True
                    for j in range(i + 1, len(file)):
                        img1name = file[j].split('/')[-1]
                        img1_camera_id = img1name.split('_')[1]
                        # img1_type = img1name.split('_')[8]
                        img1_obj_id = img1name.split('_')[0]
                        if (img0name != img1name and img1_obj_id == img0_obj_id):
                            FP += 1
                            ifexist = False
                            break
                    if (ifexist):
                        TP += 1
            else:
                taglist = []
                for i in range(len(file)):
                    taglist.append(0)
                for i in range(len(file)):
                    if (taglist[i] != 0):
                        continue
                    img0name = file[i].split('/')[-1]
                    img0_camera_id = img0name.split('_')[1]
                    # img0_type = img0name.split('_')[0]
                    img0_obj_id = img0name.split('_')[0]
                    img0 = Image.open(file[i])
                    img0 = img0.convert("RGB")
                    # img0 = img0.resize((IMG_SIZE, IMG_SIZE))
                    # img0 = PIL.ImageOps.invert(img0)
                    img0 = transform(img0)
                    img0 = np.expand_dims(img0, 0)
                    img0 = torch.from_numpy(img0).cuda()
                    # j = i+1
                    comparisonImageList = {}
                    for j in range(len(file)):
                        if (taglist[j] != 0):
                            continue
                        img1name = file[j].split('/')[-1]
                        img1_camera_id = img1name.split('_')[1]
                        # img1_type = img1name.split('_')[8]
                        img0_obj_id = img0name.split('_')[0]
                        if (img0_camera_id != img1_camera_id):
                            img1_obj_id = img1name.split('_')[1]
                            # print(("img0name:{}  img1name:{}").format(img0name,img1name))
                            img1 = Image.open(file[j])
                            img1 = img1.convert("RGB")
                            # img1 = img1.resize((IMG_SIZE, IMG_SIZE))
                            # img1 = PIL.ImageOps.invert(img1)
                            img1 = transform(img1)
                            img1 = np.expand_dims(img1, 0)
                            img1 = torch.from_numpy(img1).cuda()
                            output1, output2 = model(img0.cuda(), img1.cuda())
                            label = 1 if (img0_obj_id == img1_obj_id) else 0

                            res = F.pairwise_distance(output1, output2)
                            # print(res)
                            mindis = res if (res < mindis) else mindis
                            maxdis = res if (maxdis < res) else maxdis
                            comparisonImageList[img1name] = res

                    if (comparisonImageList == {}):
                        continue
                    list = sorted(comparisonImageList.items(), key=lambda dict: dict[1], reverse=False)
                    print(img0name, list)
                    if (float(list[0][1]) > thresold):
                        targettag = -1
                    else:
                        targettag = list[0][0].split('_')[1]
                        taglist[i] += 1
                        taglist[j] += 1
                    # targettag = list[0][0].split('_')[1]
                    # taglist[i] += 1
                    # taglist[j] += 1
                    # print(("targettag:{},  img0:{} ").format(targettag, img0_obj_id))
                    if (targettag == -1):
                        for key in comparisonImageList:
                            tag = key.split('_')[1]
                            if (img0_obj_id == tag):
                                FN += 1
                            else:
                                TN += 1
                    else:
                        for key in comparisonImageList:
                            tag = key.split('_')[1]
                            if (tag == img0_obj_id):
                                if (tag == targettag):
                                    frame0 = str(img0name).split('_')[3]
                                    frame1 = str(img1name).split('_')[3]
                                    object0 = str(img0name).split('_')[1]
                                    object1 = str(img1name).split('_')[1]
                                    bbox0 = [str(img0name).split('_')[4], str(img0name).split('_')[5],
                                             str(img0name).split('_')[6],
                                             str(img0name).split('_')[7]]
                                    bbox1 = [str(img1name).split('_')[4], str(img1name).split('_')[5],
                                             str(img1name).split('_')[6],
                                             str(img1name).split('_')[7]]
                                    camera_id0 = str(img0name).split('_')[2]
                                    camera_id1 = str(img1name).split('_')[2]
                                    addlist(reslist, frame0, object0, bbox0, camera_id0, (46, 139, 87))
                                    addlist(reslist, frame1, object1, bbox1, camera_id1, (46, 139, 87))
                                    TP += 1
                                else:
                                    FN += 1
                            else:
                                if (tag == targettag):
                                    FP += 1
                                else:
                                    TN += 1

    # print(("pre={}").format(true/count))
    accuracy = ((TP + TN) / (TP + TN + FP + FN + 1e-8)) * 100
    precision = (TP / (TP + FP + 1e-8)) * 100
    recall = (TP / (TP + FN + 1e-8)) * 100
    print("TP:{}".format(TP))
    print("TN:{}".format(TN))
    print("FP:{}".format(FP))
    print("FN:{}".format(FN))
    print(("mindis={}").format(mindis))
    print(("maxdis={}").format(maxdis))
    print("Accuracy:{}%".format(accuracy))
    print("precision:{}%".format(precision))
    print("recall:{}%".format(recall))
    with open('./result.txt', 'a') as wf:
        wf.write("test_frame网络效果" + "\n")
        wf.write("precision=" + str(precision) + "\n")
        wf.write("recall=" + str(recall) + "\n")
        wf.write("accuracy=" + str(accuracy) + "\n\n")


if __name__ == '__main__':
    config_path = "./config/test.json"  # to support different os
    with open(config_path, 'r') as f:
        config = json.load(f)
    test_dir = config['test_dir']
    thresold = int(config['thresold'])
    model_dir = config['model_dir']
    main(model_dir, thresold)
    config_path = "./config/show.json"  # to support different os
    with open(config_path, 'r') as f:
        config = json.load(f)
    gt_vis = vis.GTVisual(config['data_dir'], config['mode'])
    gt_name = os.path.split(config['data_dir'])[1] + '.json'
    gt_path = os.path.join(config['data_dir'], gt_name)
    gt_vis.show_boundingbox(gt_path, reslist)
