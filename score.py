import glob
import os
import re


def create_matrix(row,column):
    n = row
    m = column
    matrix = [None] * n
    for i in range(len(matrix)):
        matrix[i] = [0] * m
    return matrix
def update_matrix(img_name,str,matrix,xmin,ymin,xmax,ymax):
    i=xmin
    j=ymin
    xmin = int(xmin)
    ymin = int(ymin)
    xmax = int(xmax)
    ymax = int(ymax)
    for i in range(xmin,xmax):
        for j in range(ymin,ymax):
            print("img_name:.{},分数表：{},xmin:{},ymin：{},xmax：{},ymax:{}".format(img_name,str,xmin, ymin, xmax, ymax))
            print('x={},y={}'.format(i,j))
            if(i> 1919):
                i = 1919
            if(j> 1207):
                j = 1207
            matrix[i][j] += 1
    return matrix

def create_excel(matrix,excel_name):
    output = open('../data/pic/{}.xls'.format(excel_name), 'w+', encoding='gbk')
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            output.write(str(matrix[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            output.write('\t')  # 相当于Tab一下，换一个单元格
        output.write('\n')  # 写完一行立马换行
    output.close()

if __name__ == "__main__":
    data_dir = '../data/pic/test/'
    score12_1 = create_matrix(1920, 1208)
    score12_2 = create_matrix(1920, 1208)
    score23_2 = create_matrix(1920, 1208)
    score23_3 = create_matrix(1920, 1208)
    score34_3 = create_matrix(1920, 1208)
    score34_4 = create_matrix(1920, 1208)
    i = 0
    for i in range(1201):
        img_camera1_list = []
        img_camera2_list = []
        img_camera3_list = []
        img_camera4_list = []
        img_path_list = glob.glob(data_dir + str(i) + '/*' + 'jpg')  # 返回所有匹配文件路径列表
        for img_path in img_path_list:
            img_name = os.path.basename(img_path)
            img_name_camera = re.findall(r"\d+", img_name)[2]
            if (int(img_name_camera) == 1):
                img_camera1_list.append(img_name)
            elif (int(img_name_camera) == 2):
                img_camera2_list.append(img_name)
            elif (int(img_name_camera) == 3):
                img_camera3_list.append(img_name)
            elif (int(img_name_camera) == 4):
                img_camera4_list.append(img_name)
            else:
                print("False!Can't match camera!")
            for img_name1 in img_camera1_list:
                for img_name2 in img_camera2_list:
                    img_name1_object = re.findall(r"\d+", img_name1)[1]
                    img_name2_object = re.findall(r"\d+", img_name2)[1]
                    print(img_name1,img_name2)
                    if(img_name1_object==img_name2_object):
                        img_name1_xmin = re.findall(r"\d+", img_name1)[4]
                        img_name1_ymin = re.findall(r"\d+", img_name1)[5]
                        img_name1_xmax = re.findall(r"\d+", img_name1)[6]
                        img_name1_ymax = re.findall(r"\d+", img_name1)[7]
                        score12_1 = update_matrix(img_name1,'score12_1',score12_1, img_name1_xmin, img_name1_ymin, img_name1_xmax, img_name1_ymax)
                        img_name2_xmin = re.findall(r"\d+", img_name2)[4]
                        img_name2_ymin = re.findall(r"\d+", img_name2)[5]
                        img_name2_xmax = re.findall(r"\d+", img_name2)[6]
                        img_name2_ymax = re.findall(r"\d+", img_name2)[7]
                        score12_2 = update_matrix(img_name2,'score12_2',score12_2, img_name2_xmin, img_name2_ymin, img_name2_xmax, img_name2_ymax)
            for img_name2 in img_camera2_list:
                for img_name3 in img_camera3_list:
                    img_name2_object = re.findall(r"\d+", img_name2)[1]
                    img_name3_object = re.findall(r"\d+", img_name3)[1]
                    print(img_name2, img_name3)
                    if(img_name2_object==img_name3_object):
                        img_name2_xmin = re.findall(r"\d+", img_name2)[4]
                        img_name2_ymin = re.findall(r"\d+", img_name2)[5]
                        img_name2_xmax = re.findall(r"\d+", img_name2)[6]
                        img_name2_ymax = re.findall(r"\d+", img_name2)[7]
                        score23_2 = update_matrix(img_name2,'score23_2',score23_2, img_name2_xmin, img_name2_ymin, img_name2_xmax, img_name2_ymax)
                        img_name3_xmin = re.findall(r"\d+", img_name3)[4]
                        img_name3_ymin = re.findall(r"\d+", img_name3)[5]
                        img_name3_xmax = re.findall(r"\d+", img_name3)[6]
                        img_name3_ymax = re.findall(r"\d+", img_name3)[7]
                        score23_3 = update_matrix(img_name3,'score23_3',score23_3, img_name3_xmin, img_name3_ymin, img_name3_xmax, img_name3_ymax)
            for img_name3 in img_camera3_list:
                for img_name4 in img_camera4_list:
                    img_name3_object = re.findall(r"\d+", img_name3)[1]
                    img_name4_object = re.findall(r"\d+", img_name4)[1]
                    print(img_name3, img_name4)
                    if(img_name3_object==img_name4_object):
                        img_name3_xmin = re.findall(r"\d+", img_name3)[4]
                        img_name3_ymin = re.findall(r"\d+", img_name3)[5]
                        img_name3_xmax = re.findall(r"\d+", img_name3)[6]
                        img_name3_ymax = re.findall(r"\d+", img_name3)[7]
                        score34_3 = update_matrix(img_name3,'score34_3',score34_3, img_name3_xmin, img_name3_ymin, img_name3_xmax, img_name3_ymax)
                        img_name4_xmin = re.findall(r"\d+", img_name4)[4]
                        img_name4_ymin = re.findall(r"\d+", img_name4)[5]
                        img_name4_xmax = re.findall(r"\d+", img_name4)[6]
                        img_name4_ymax = re.findall(r"\d+", img_name4)[7]
                        score34_4 = update_matrix(img_name4,'score34_4',score34_4, img_name4_xmin, img_name4_ymin, img_name4_xmax, img_name4_ymax)

        create_excel(score12_1, 'score12_1')
        create_excel(score12_2, 'score12_2')
        create_excel(score23_2, 'score23_2')
        create_excel(score23_3, 'score23_3')
        create_excel(score34_3, 'score34_3')
        create_excel(score34_4, 'score34_4')