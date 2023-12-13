import cv2
import os
import numpy as np
import torch

import torch.nn as nn
from tqdm import tqdm
#from scipy.ndimage.morphology import distance_transform_edt
from networks.exchange_dlink34net import DinkNet34_CMMPNet
from networks.dlinknet import DinkNet34, LinkNet34
from networks.deeplabv3plus import DeepLabV3Plus
from networks.unet import Unet
from networks.resunet import ResUnet, ResUnet1DConv
#from networks.dlinknet34 import DinkNet34_CMMPNet
# 源目录
sat_root = 'E:/ML_data/remote_data/BJRoad/test/image/'
gps_root= 'E:/ML_data/remote_data/BJRoad/test/gps/'
test_mask_dir='E:/ML_data/remote_data/BJRoad/test/mask/'
# 输出目录
outPath ='E:/ML_data/remote_data/BJRoad/test/ablation/allpol1/resave/'

image_id  = [x[:-9] for x in os.listdir(test_mask_dir) if x.find('mask.png') != -1]



def get_model(model_name):
    if model_name == 'CMMPNet':
        model = DinkNet34_CMMPNet(bn=0.02)
    else:
        print("[ERROR] can not find model ", model_name)
        assert(False)
    return model
net = get_model('CMMPNet')  #exchange_pol1_epoch28_val0.7234_test0.6384
state_dict = torch.load('save_model/ablation/allpol1_epoch30_val0.7119_test0.6332.pth', map_location=torch.device('cpu'))
net.load_state_dict(state_dict)

n=0
for name, param in net.named_parameters():
    if param.requires_grad and name.endswith('weight') and 'bn2' in name:
        a=param.detach()
        n=n+1
        print('第n层：',n)
        # print(a)
        # if n==25 or n==26:
        print(list(np.array(a)))


# new_state = {}
# for key, value in state_dict.items():
#     new_state[key.replace('module.', '')] = value
# net.load_state_dict(new_state)
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

    pred = np.squeeze(pred, axis=0).transpose(1, 2, 0)*255
    cv2.imwrite(destsource + name+'.png', pred)

if __name__ == '__main__':
    print('start')
    # image_id=['5_21','8_26','11_9','38_26','53_0']
    for i in image_id:
        img_path = os.path.join(sat_root, "{0}_sat.{1}").format(i, "png")
        gps_path = os.path.join(gps_root, "{0}_gps.{1}").format(i, "jpg")

        processImage(img_path,gps_path, net,outPath, i)
