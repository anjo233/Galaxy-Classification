import os
import json
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import random

from model import convnext_base as create_model

plot_num=5 #随机选择数目
TEST_IMGES_PATH = r'D:\ProgramCode\PythonFile\galaxy\efficientnetv2\Galaxy_class\Train_images'
Pre_IMGES_PATH = r'D:\ProgramCode\PythonFile\galaxy\ConvNeXt\predict_file'
json_path = r'galaxy\ConvNeXt\class_indices.json'
model_weight_path = r"D:\ProgramCode\PythonFile\galaxy\ConvNeXt\weights\best_model-29.pth"
TEST_ID_ClASS = r'D:\ProgramCode\PythonFile\galaxy\ConvNeXt\Test_name.csv'
num_classes = 5
img_size = 224
data_transform = transforms.Compose([transforms.CenterCrop(img_size),#中心裁剪图片
                                   transforms.RandomRotation(degrees = 30),
                                   transforms.RandomHorizontalFlip(),#随机水平翻转图片
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                   ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using {device} device.")

images_messgae = []

test_solutions = pd.read_csv(TEST_ID_ClASS)
test_id_list = test_solutions.iloc[:,0]
test_class_list = test_solutions.iloc[:,1]




# read class_indict
assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
with open(json_path, "r") as f:
    class_indict = json.load(f)


# create model
model = create_model(num_classes=num_classes).to(device)
# load model weights

model.load_state_dict(torch.load(model_weight_path, map_location=device))
model.eval()


def main(id_list,class_list):

    # load image
    for i in range(len(id_list)):
        img_pre_dick = {}

        img_path_c = TEST_IMGES_PATH + "\\"+ class_indict[str(class_list[i])] +"\\"+ id_list[i]
        
        assert os.path.exists(img_path_c), "file: '{}' dose not exist.".format(id_list[i])
        img = Image.open(img_path_c)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        
        with torch.no_grad():
        # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        

        img_pre_dick['id'] = id_list[i]
        img_pre_dick['R_class'] = "class_{}".format(class_list[i])  #真实类别值为（0，1，2，3，4）也可以写成class_indict[str(class_list[i])]
        img_pre_dick['P_class'] = "class_{}".format(predict_cla)  #预测类别值为（0，1，2，3，4）也可以写成class_indict[str(predict_cla)]
        img_pre_dick['Prob'] = predict[predict_cla].numpy()
        img_pre_dick['spot_on'] = 1 if class_list[i] == predict_cla else 0
        # print_res = "id:{} \n class: {} \n  prob: {:.3}".format(id_list[i],class_indict[str(predict_cla)],predict[predict_cla].numpy())
        
        for i in range(len(predict)):
            img_pre_dick["class_{}_p".format(i)] = predict[i].numpy()
            #print("class_{}_p: {:10}    prob: {:.3}".format(i,class_indict[str(i)],predict[i].numpy()))
        
        images_messgae.append(img_pre_dick)
    print(images_messgae[:5])
    return images_messgae

def OUT_csv(dick):
    df = pd.DataFrame(dick)
    df.to_csv(r'D:\ProgramCode\PythonFile\galaxy\ConvNeXt\all_output.csv', index=False)    


def Random_img(path):
    files_id = os.listdir(path)
    random_image = random.sample(files_id,plot_num)
    return random_image

def plot_img_prodict(imgid_path):

    plt.figure(figsize=(16,5))
    for i in range(plot_num):

        img_path_c = Pre_IMGES_PATH + "\\" + imgid_path[i]
        plt.subplot(1,plot_num,i+1)
        assert os.path.exists(img_path_c), "file: '{}' dose not exist.".format(imgid_path[i])
        img = Image.open(img_path_c)
        plt.imshow(img)
        # [N, C, H, W]
        img = data_transform(img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        
        with torch.no_grad():
        # predict class
            output = torch.squeeze(model(img.to(device))).cpu()
            predict = torch.softmax(output, dim=0)
            predict_cla = torch.argmax(predict).numpy()
        
        print_res = "id:{} \n class: {} \n  prob: {:.3}".format(imgid_path[i],
                                                class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())

        plt.title(print_res)
        print(imgid_path[i])
        for i in range(len(predict)):
            print("class: {:10}    prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
        plt.axis(False) 

        
    plt.subplots_adjust(wspace=0.2, hspace=0.2)
    plt.show()


    

            

        

if __name__ == '__main__':
    #
    # pre_imges_list = Random_img(Pre_IMGES_PATH)
    # plot_img_prodict(pre_imges_list)

    images_mess = main(test_id_list,test_class_list)
    OUT_csv(images_mess)