import torch
import numpy as np
import time
import torch.nn as nn
from torch.nn import init
import os
import sys
import cv2
from utils.model_init import model_init
from framework import Framework
from utils.datasets import prepare_Beijing_dataset, prepare_TLCGIS_dataset


#训练cmx用的
from models.builder import EncoderDecoder as segmodel
from easydict import EasyDict as edict
C = edict()
config = C
C.backbone = 'mit_b2' # Remember change the path below.
C.pretrained_model = 'pre-trained_weights/mit_b2.pth'
C.decoder = 'MLPDecoder'
C.decoder_embed_dim = 512
C.optimizer = 'AdamW'
C.fix_bias = True
C.bn_eps = 1e-3
C.bn_momentum = 0.1
C.num_classes = 1

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
 
    def flush(self):
        pass


def get_dataloader(args):
    if args.dataset =='BJRoad':
        train_ds, val_ds, test_ds = prepare_Beijing_dataset(args) 
    elif args.dataset == 'TLCGIS' or args.dataset.find('Porto') >= 0:
        train_ds, val_ds, test_ds = prepare_TLCGIS_dataset(args) 
    else:
        print("[ERROR] can not find dataset ", args.dataset)
        assert(False)  

    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=True,  drop_last=False)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False, drop_last=False)
    test_dl  = torch.utils.data.DataLoader(test_ds,  batch_size=BATCH_SIZE, num_workers=args.workers, shuffle=False, drop_last=False)
    return train_dl, val_dl, test_dl


def train_val_test(args):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    net = segmodel(cfg=config, criterion=criterion, norm_layer=nn.BatchNorm2d)
    print('lr:',args.lr)
    optimizer = torch.optim.Adam(params=net.parameters(), lr=args.lr)
    # 多gpu得到的模型dict前面会加module
#     new_state = {} 
#    state_dict = torch.load('save_model/edge_val0.6420_test0.6098.pth')#, map_location=torch.device('cpu'))
#     for key, value in state_dict.items():
#         new_state[key.replace('module.', '')] = value
#     net.load_state_dict(new_state)
#     net.load_state_dict(state_dict)

    framework = Framework(net, optimizer, dataset=args.dataset)
    
    train_dl, val_dl, test_dl = get_dataloader(args)
    framework.set_train_dl(train_dl)
    framework.set_validation_dl(val_dl)
    framework.set_test_dl(test_dl)
    framework.set_save_path(WEIGHT_SAVE_DIR)

    framework.fit(epochs=args.epochs,lam=2e-4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='CMMPNet')
    parser.add_argument('--lr',    type=float, default=2e-4)
    parser.add_argument('--name',  type=str, default='')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--sat_dir', type=str, default='/home/imi432004/BJRoad/BJRoad/train_val/image')
    parser.add_argument('--mask_dir', type=str, default='/home/imi432004/BJRoad/BJRoad/train_val/mask')
    parser.add_argument('--gps_dir', type=str, default='/home/imi432004/BJRoad/BJRoad/train_val/gps')
    parser.add_argument('--edge_dir', type=str, default='/home/imi432004/BJRoad/BJRoad/train_val/edge')
    parser.add_argument('--test_sat_dir', type=str, default='/home/imi432004/BJRoad/BJRoad/test/image')
    parser.add_argument('--test_mask_dir', type=str, default='/home/imi432004/BJRoad/BJRoad/test/mask')
    parser.add_argument('--test_gps_dir', type=str, default='/home/imi432004/BJRoad/BJRoad/test/gps')
    # parser.add_argument('--sat_dir',  type=str, default=r'E:\ML_data\remote_data\BJRoad\train_val\image')
    # parser.add_argument('--mask_dir', type=str, default=r'E:\ML_data\remote_data\BJRoad\train_val\mask')
    # parser.add_argument('--gps_dir',  type=str, default=r'E:\ML_data\remote_data\BJRoad\train_val\gps')
    # parser.add_argument('--edge_dir', type=str, default=r'E:\ML_data\remote_data\BJRoad\train_val\edge')
    parser.add_argument('--test_edge_dir', type=str, default='/home/imi432004/BJRoad/BJRoad/test/edge')
    # parser.add_argument('--test_sat_dir',  type=str, default=r'E:\ML_data\remote_data\BJRoad\test\image')
    # parser.add_argument('--test_mask_dir', type=str, default=r'E:\ML_data\remote_data\BJRoad\test\mask')
    # parser.add_argument('--test_gps_dir',  type=str, default=r'E:\ML_data\remote_data\BJRoad\test\gps')
    # parser.add_argument('--connect_dir1',  type=str, default=r'F:\ML_data\remote_data\BJRoad\train_val\connect_8_d1')
    parser.add_argument('--connect_dir2', type=str, default=r'F:\ML_data\remote_data\BJRoad\train_val\connect_8_d3')
    parser.add_argument('--test_connect_dir1', type=str,default=r'F:\ML_data\remote_data\BJRoad\test\connect_8_d1')
    parser.add_argument('--test_connect_dir2', type=str,default=r'F:\ML_data\remote_data\BJRoad\test\connect_8_d3')
    parser.add_argument('--lidar_dir',  type=str, default='/home/imi432004/porto_dataset/gps')
    parser.add_argument('--split_train_val_test', type=str, default='/home/imi432004/porto_dataset/split_5')
    parser.add_argument('--weight_save_dir', type=str, default='./save_model')
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--use_gpu',  type=bool, default=True)
    parser.add_argument('--gpu_ids',  type=str, default='0')
    parser.add_argument('--workers',  type=int, default=0)
    parser.add_argument('--epochs',  type=int, default=30)
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--dataset', type=str, default='BJRoad')
    parser.add_argument('--down_scale', type=bool, default=False)
    args = parser.parse_args()

    if args.use_gpu:
        try:
            gpu_list = [int(s) for s in args.gpu_ids.split(',')]
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
        BATCH_SIZE = args.batch_size * len(gpu_list)
    else:
        BATCH_SIZE = args.batch_size
        
    WEIGHT_SAVE_DIR = os.path.join(args.weight_save_dir, f"{args.model}_{args.dataset}_"+time.strftime("%Y-%m-%d_%H_%M_%S", time.localtime())+"/")
    if not os.path.exists(WEIGHT_SAVE_DIR):
        os.makedirs(WEIGHT_SAVE_DIR)
    print("Log dir: ", WEIGHT_SAVE_DIR)
    
    path = os.path.abspath(os.path.dirname(__file__))
    type = sys.getfilesystemencoding()
    sys.stdout = Logger(WEIGHT_SAVE_DIR+'train.log')
    torch.manual_seed(114514)
    torch.cuda.manual_seed(114514)
    train_val_test(args)
    print("[DONE] finished")

