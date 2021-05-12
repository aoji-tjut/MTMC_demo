# -*- coding: UTF-8 -*-
import os
import json
import cv2
import numpy as np
import pdb
import PIL.Image as img
import vis
import tools.trainmodel as train
import tools.testbyframe as test
import torch
import torchvision


if __name__ == '__main__':
    # config_path = "./config/vis.json"  # to support different os
    # with open(config_path, 'r') as f:
    #     config = json.load(f)
    # gt_vis = vis.GTVisual(config['data_dir'], config['mode'])
    # gt_name = os.path.split(config['data_dir'])[1] + '.json'
    # gt_path = os.path.join(config['data_dir'], gt_name)
    # gt_vis.show_result(gt_path)
    #
    # model = train.train()
    # torch.save(model.state_dict(), '../model/model.pt')
    # print('Model saved successfully')

    test.main()

    config_path = "./config/show.json"  # to support different os
    with open(config_path, 'r') as f:
        config = json.load(f)
    gt_vis = vis.GTVisual(config['data_dir'], config['mode'])
    gt_name = os.path.split(config['data_dir'])[1] + '.json'
    gt_path = os.path.join(config['data_dir'], gt_name)
    gt_vis.show_boundingbox(gt_path, test.reslist)
    # model.load_state_dict(torch.load('./model/model.pt'))
