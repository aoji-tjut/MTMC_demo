import glob
import os
import cv2
import shutil
import re
import json

wide = 1920
high = 1208

def main(data_dir,out_dir):
    i = 0
    for i in range(1201):
        img_path_list = glob.glob(data_dir + str(i) + '/*' + 'jpg')  # 返回所有匹配文件路径列表
        img_camera1_list = []
        img_camera2_list = []
        img_camera3_list = []
        img_camera4_list = []
        img_save_list = []
        for img_path in img_path_list:
            img_name = os.path.basename(img_path)  # 返回图像名称
            img_name_camera = re.findall(r"\d+", img_name)[2]  # 利用正则表达式从图像名称中提取相机号
            if (int(img_name_camera) == 1):
                img_camera1_list.append(img_name)
                # print("第{}号帧的1号相机列表添加成功".format(i))
            elif (int(img_name_camera) == 2):
                img_camera2_list.append(img_name)
                # print("第{}号帧的2号相机列表添加成功".format(i))
            elif (int(img_name_camera) == 3):
                img_camera3_list.append(img_name)
                # print("第{}号帧的3号相机列表添加成功".format(i))
            elif (int(img_name_camera) == 4):
                img_camera4_list.append(img_name)
                # print("第{}号帧的4号相机列表添加成功".format(i))
            else:
                print("False!Can't match camera!")
        path = out_dir + str(i) + "/1&2/"
        if not os.path.exists(path):
            os.makedirs(path)
        for img_name_1 in img_camera1_list:
            for img_name_2 in img_camera2_list:
                if img_name_1.split('_')[8] == img_name_2.split('_')[8]:
                    print("判别类型是：{}和{}".format(img_name_1.split('_')[8], img_name_2.split('_')[8]))
                    img_name_1_left = int(re.findall(r"\d+", img_name_1)[4])
                    img_name_2_right = int(re.findall(r"\d+", img_name_2)[6])
                    if (img_name_1_left <= 940) and (img_name_2_right >= 1050 and img_name_2_right <= 1800):
                        shutil.copy(data_dir + str(i) + "/" + img_name_1, path + img_name_1)
                        shutil.copy(data_dir + str(i) + "/" + img_name_2, path + img_name_2)
                        img_save_list.append(img_name_1)
                        img_save_list.append(img_name_2)
        path = out_dir + str(i) + "/2&3/"
        if not os.path.exists(path):
            os.makedirs(path)
        for img_name_2 in img_camera2_list:
            for img_name_3 in img_camera3_list:
                if img_name_2.split('_')[8] == img_name_3.split('_')[8]:
                    print("判别类型是：{}和{}".format(img_name_2.split('_')[8], img_name_3.split('_')[8]))
                    img_name_2_left = int(re.findall(r"\d+", img_name_2)[4])
                    img_name_3_right = int(re.findall(r"\d+", img_name_3)[6])
                    if (img_name_2_left <= 1460) and (img_name_3_right >= 990):
                        shutil.copy(data_dir + str(i) + "/" + img_name_2, path + img_name_2)
                        shutil.copy(data_dir + str(i) + "/" + img_name_3, path + img_name_3)
                        img_save_list.append(img_name_2)
                        img_save_list.append(img_name_3)
        path = out_dir + str(i) + "/3&4/"
        if not os.path.exists(path):
            os.makedirs(path)
        for img_name_3 in img_camera3_list:
            for img_name_4 in img_camera4_list:
                if img_name_3.split('_')[8] == img_name_4.split('_')[8]:
                    print("判别类型是：{}和{}".format(img_name_3.split('_')[8], img_name_4.split('_')[8]))
                    img_name_3_left = int(re.findall(r"\d+", img_name_3)[4])
                    img_name_4_right = int(re.findall(r"\d+", img_name_4)[6])
                    if (img_name_3_left >= 160 and img_name_3_left <= 680) and (img_name_4_right >= 1045):
                        shutil.copy(data_dir + str(i) + "/" + img_name_3, path + img_name_3)
                        shutil.copy(data_dir + str(i) + "/" + img_name_4, path + img_name_4)
                        img_save_list.append(img_name_3)
                        img_save_list.append(img_name_4)
        for img_path in img_path_list:
            img_name = os.path.basename(img_path)
            if not os.path.exists(out_dir + str(i) + "/others/"):
                os.makedirs(out_dir + str(i) + "/others/")
            if img_name in img_save_list:
                print("已经存在了！")

            else:
                path = out_dir + str(i) + "/others/"

                shutil.copy(data_dir + str(i) + "/" + img_name, path + img_name)

if __name__ == "__main__":
    config_path = "./config/add.json"  # to support different os
    with open(config_path, 'r') as f:
        config = json.load(f)
    data_dir = config["data_dir"]
    out_dir = config["out_dir"]
    main(data_dir,out_dir)









