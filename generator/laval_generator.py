
import numpy as np
import os
import tensorflow as tf
import cv2
from PIL import Image
from utils import tonemap
class LavalGenerator(tf.keras.utils.Sequence):
    def __init__(self, args, mode='train'):
        self.args = args
        root_dir = args.dataset
        batch_size = args.batch_size
        augment = args.augment
        img_size = args.img_size

        self.resize_size = ( int(img_size/0.875),  int(img_size/0.875))
        self.crop_size = (img_size, img_size)
        #确定输入图片路径和标签的路径
        self.data_dir = '/root/LavalWarp/hdrInputs'
        self.label_dir = '/root/bjy/SHNet/SHDataset/'
        self.root_dir = os.path.join(root_dir, mode)
        self.img_path_list = []
        self.label_path_list = []
        with open('/root/bjy/SHNet/SHDataset/namelist/warped_laval_train.txt', 'r') as readfile:
            lines = readfile.readlines()
            for line in lines:
                line = line.strip()
                im_path = self.data_dir + "/" + line + ".exr"
                self.img_path_list.append(im_path)
                label_path = line# + "_sh_light_l10.exr.npy"
                self.label_path_list.append(label_path)
            #self.img_path_list = self.img_path_list[:1000]
            #self.label_path_list = self.label_path_list[:1000]
        if mode == 'train':
            #随机扩充使得可以整除batch
            pad_len = len(self.img_path_list)%batch_size
            img_path_list_len = len(self.img_path_list)
            for _ in range(pad_len):
                rand_index = np.random.randint(0,img_path_list_len)
                self.img_path_list.append(self.img_path_list[rand_index])
                self.label_path_list.append(self.label_path_list[rand_index])
            self.data_index = np.arange(0, len(self.label_path_list))
            np.random.shuffle(self.data_index)
        else:
            
            self.data_index = np.arange(0, len(self.label_path_list))
        self.img_path_list = np.array(self.img_path_list)
        self.label_path_list = np.array(self.label_path_list)
        self.augment = augment
        self.mode = mode
        self.batch_size = batch_size
        self.eppch_index = 0

        self.tone = tonemap.TonemapHDR(gamma=2.4, percentile=50, max_mapping=0.5)
    
    def read_img(self, path):
        #读取图片，并且tonemapping到0，1之间
        image = cv2.imread(path, -1)
        
        img, alpha = self.tone(image)
        return img
        # return image[:, :, ::-1]
    def on_epoch_end(self):
        if self.mode == 'train':
            np.random.shuffle(self.data_index)
        self.eppch_index += 1
    def __len__(self):
        return int(np.ceil(len(self.img_path_list) / self.batch_size))
    def __getitem__(self, batch_index):
        #每个patch进行计算
        batch_img_paths = self.img_path_list[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        batch_labels_path = self.label_path_list[self.data_index[batch_index * self.batch_size:(batch_index + 1) * self.batch_size]]
        batch_labels10 = []
        batch_labels1 = []
        batch_labels2 = []
        batch_imgs = []
        for i in range(len(batch_img_paths)):
            img = self.read_img(batch_img_paths[i])
            batch_imgs.append(img)
            #保存img
            #img = img * 255
            #img = img.astype('int')
            #cv2.imwrite("./dataset/WarpedTone/" + batch_labels_path[i] + ".png", img)
            
            max_num = 1.5
            min_num = -0.7
            data1 = np.load(self.label_dir + "hdroutput1/" + batch_labels_path[i] + "_sh_light_l1.exr.npy")
            batch_labels1.append(np.reshape(data1, -1))
            

            data2 = np.load(self.label_dir + "hdroutput2/" + batch_labels_path[i] + "_sh_light_l2.exr.npy")
            batch_labels2.append(np.reshape(data2, -1))
            
            data10 = np.load(self.label_dir + "hdroutput10/" + batch_labels_path[i] + "_sh_light_l10.exr.npy")
            batch_labels10.append(np.reshape(data10, -1))
            
            '''范围check
            if np.min(data1) < min_num or np.max(data1) > max_num:
                print(data1,"the data")
                assert(1==0)
            if np.min(data2) < min_num or np.max(data2) > max_num:
                print(data2,"the data")
                assert(1==0)
            if np.min(data10) < min_num or np.max(data10) > max_num:
                print(data10,"the data")
                assert(1==0)'''
        #assert(1==0)
        batch_imgs = np.array(batch_imgs)
        batch_labels1 = np.array(batch_labels1)
        batch_labels2 = np.array(batch_labels2)
        batch_labels10 = np.array(batch_labels10)
        

        return batch_imgs, [batch_labels1, batch_labels2, batch_labels10]




