import cv2
import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.ndimage.morphology import distance_transform_edt
#from networks.exchange_dlink34net import DinkNet34_CMMPNet
from networks.CMMPNet import DinkNet34_CMMPNet
#from networks.dlinknet import DinkNet34, LinkNet34
from networks.dlinknet34 import DinkNet34_CMMPNet
# from networks.hrnet import hrnet18
# 源目录
sat_root = 'E:/ML_data/remote_data/porto_dataset/rgb'
gps_root= 'E:/ML_data/remote_data/porto_dataset/gps'
# 输出目录
outPath = 'E:/ML_data/remote_data/porto_dataset/dlinknet_pred/resave/'
split_train_val_test ='E:/ML_data/remote_data/porto_dataset/split_5'

train_list = val_list = test_list = []
with open(os.path.join(split_train_val_test, 'train.txt'), 'r') as f:
    train_list = [x[:-1] for x in f]
with open(os.path.join(split_train_val_test, 'valid.txt'), 'r') as f:
    val_list = [x[:-1] for x in f]
with open(os.path.join(split_train_val_test, 'test.txt'), 'r') as f:
    test_list = [x[:-1] for x in f]

def get_model(model_name):
    if model_name == 'CMMPNet':
        model = DinkNet34_CMMPNet()
    else:
        print("[ERROR] can not find model ", model_name)
        assert(False)
    return model
net = get_model('CMMPNet')
new_state = {}
state_dict = torch.load(r'C:\Users\lenovo\Desktop\exchange\sun\save_model\porto5\dlinknet_val0.7041_testgiou0.7041.pth', map_location=torch.device('cpu'))
# for key, value in state_dict.items():
#     new_state[key.replace('module.', '')] = value
# net.load_state_dict(new_state)
net.load_state_dict(state_dict)
def processImage(path1, path2,net,destsource, name):
    net.eval()
    img = cv2.imread(path1)
    img1 = cv2.imread(path2, cv2.IMREAD_GRAYSCALE)
    print(name)
    img1 = np.expand_dims(img1, axis=2)
    img = np.concatenate([img, img1], axis=2)
    img = cv2.resize(img, (512, 512))
    img = np.array(img, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    img = img[np.newaxis, :, :, :]
    img = torch.tensor(img).cuda()
    net.cuda()
    with torch.no_grad():
        pred = net.forward(img)
    pred = pred.cpu().numpy()
    pred = np.squeeze(pred, axis=0).transpose(1, 2, 0)*255.0
    cv2.imwrite(destsource + name+'.png', pred)

if __name__ == '__main__':
    for i in test_list:
        img_path = os.path.join(sat_root, "{0}.{1}").format(i, "png")
        gps_path = os.path.join(gps_root, "{0}.{1}").format(i, "png")
        processImage(img_path,gps_path, net,outPath, i)
