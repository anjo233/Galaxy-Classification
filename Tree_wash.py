import pandas as pd

import numpy as np

import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import random
import shutil
from time import time
# from torchtoolbox.transform import Cutout
import os
#os.environ['KMP_DUPLICATE_LIB_OK']='True'
from PIL import Image
TRAIN_DATA_DIR = r'D:\ProgramCode\PythonFile\galaxy\galaxynet\images_training_rev1'
TRAIN_TARGET_CSV = r'D:\ProgramCode\PythonFile\galaxy\galaxynet\training_solutions_rev1.csv'

CLASS_TRAIN = r'D:\ProgramCode\PythonFile\galaxy\galaxynet\Galaxy_class\Train_images'
CLASS_TEST = r'D:\ProgramCode\PythonFile\galaxy\galaxynet\Galaxy_class\Test_images'
CLASS_TARGET_CSV = r'D:\ProgramCode\PythonFile\galaxy\efficientnetv2\Galaxy_class'

train_solutions = pd.read_csv(TRAIN_TARGET_CSV, header=None)

preprocess = transforms.Compose([
        transforms.RandomRotation(degrees = 30),
        #transforms.CenterCrop(224),
        #transforms.RandomResizedCrop(224,scale=(0.8, 1.1)),
        
        #transforms.Resize(int(224 * 1.143)),
        #transforms.RandomAutocontrast(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])#正则化
            

                                     
                                    
                                     
train_solutions.head()
datatrain = train_solutions.values


class_lits =list( datatrain[0,1:])#类别列表
imges_id = datatrain[1:,0]#图片的ID
set_data = datatrain[1:,1:]#去掉标签后的全部的数据
set_data.astype(float)
#print(set_data, type(set_data[0,7]))
galaxy_num = len(imges_id)
class_name_id = ["Cigar-shaped smooth","In between smooth","Completely Round smooth","Edge-on","Spiral"]
CSV_HEADER = [ "Class0", "Class1", "Class2", "Class3", "Class4"]
Class_list = []
class_num_lits= [0,0,0,0,0]
#分类函数
def Tree_galaxy(dick_list,i):
    imges_class = {}
    if float(dick_list[class_lits.index('Class1.1')]) >= 0.469:
        if float(dick_list [class_lits.index('Class7.1')]) >= 0.5:
            imges_class["id"]=imges_id[i]
            imges_class["class"]=("Completely Round smooth")
            class_num_lits[0] += 1
        elif float(dick_list [class_lits.index('Class7.2')]) >= 0.5:
            imges_class["id"]=imges_id[i]
            imges_class["class"]=("In between smooth")
            class_num_lits[1] += 1
        elif float(dick_list[class_lits.index('Class7.3')]) >= 0.5:
            imges_class["id"]=imges_id[i]
            imges_class["class"]=("Cigar-shaped smooth")
            class_num_lits[2] += 1
    elif float(dick_list[class_lits.index('Class1.2')]) >= 0.430:
        if float(dick_list[class_lits.index('Class2.1')]) >= 0.602:
            imges_class["id"]=imges_id[i]
            imges_class["class"]=("Edge-on")
            class_num_lits[3] += 1
        elif float(dick_list[class_lits.index('Class2.2')]) >= 0.715 and float(dick_list[class_lits.index('Class4.1')]) >= 0.619:
            imges_class["id"]=imges_id[i]
            imges_class["class"]=("Spiral")
            class_num_lits[4] += 1
    
    return imges_class,class_num_lits

#显示函数
def plot_galaxy(path, class_list):
    #random_image=random.sample(os.listdir(path),sample)
    plot_num = 5
    # random_image = random.sample(class_list,plot_num)
    random_image = class_list[:plot_num]
    #print(random_image)
    class_list_id = []
    for item in random_image:
        class_list_id.append(item['id'])
    # print(class_list_id)
    plt.figure(figsize=(16,5))
    for i in range(plot_num):
        plt.subplot(1,plot_num,i+1)
        img=Image.open(os.path.join(path,str(class_list_id[i]) + ".jpg"))
        # 对图像进行预处理
        img_tensor = preprocess(img)
        # 将 img_tensor 的形状转换为 (height, width, channel)
        img_tensor = torch.transpose(img_tensor, 0, 2).numpy()
        # img_tensor = img_tensor*-1
        
        #print(img_tensor[1])
        plt.imshow(img_tensor)
        plt.title(f'imges_id: {class_list_id[i]}\nShape: {img_tensor.shape}\nClass: {random_image[i]["class"]}')
        plt.axis(False) 
    
    plt.show()

output_csv_list = []
def label_CSV(dick,out_path) :
    
    
    for item in dick:
        lab_list = []
        lab_list.append(item['id'])
        lab_list.append(CSV_HEADER[class_name_id.index(item['class'])])

        output_csv_list.append(lab_list)

    df = pd.DataFrame(output_csv_list)
    df.to_csv(out_path + '/train_label.csv', index=False,header=False, encoding='utf-8-sig', mode='w', line_terminator='\n', sep=',')
    return output_csv_list

def Copy_files(file_list, dest_folder,imgs_folder):
    for item in file_list:
        source_path = os.path.abspath(os.path.join(imgs_folder, str(item['id'])+'.jpg'))
        dest_path = os.path.join(dest_folder+'\\'+item['class'], str(item['id'])+'.jpg')
        shutil.copy(source_path, dest_path)
    

for i in range(20):
    
    imges_p = list(set_data[i])
    imges_class,class_num_lits = Tree_galaxy(imges_p,i)
    
    if imges_class != {}:
        Class_list.append(imges_class)
        # print(imges_class)
  
#print(Class_list)
#print(class_num_lits,sum(class_num_lits))
output_csv_list = label_CSV(Class_list,CLASS_TARGET_CSV)
print(output_csv_list)
#plot_galaxy(TRAIN_DATA_DIR, Class_list)
# Copy_files(Class_list, CLASS_TRAIN, TRAIN_DATA_DIR)

'''
for i in range(test_num):
    
    test_imges_p = list(test_set_data[i])
    imges_class = Tree_galaxy(test_imges_p,i)
    if imges_class != {}:
        Class_list_Test.append(imges_class)
        print(imges_class)
print(Class_list_Test)
plot_galaxy(TEST_DATA_DIR, Class_list_Test)
Copy_files(Class_list_Test, CLASS_TEST, TEST_DATA_DIR)
'''