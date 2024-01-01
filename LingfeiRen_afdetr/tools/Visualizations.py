import matplotlib.pyplot as plt
import torch

import torchvision
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# from utils.imgname import read_img_name
import seaborn as sns


import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def read_img_name():
    f = open(r"./imgname.txt", "r")
    file = f.readlines()
    for each in file:
        each = each.strip('\n')
        return each

def network_inputs_visual(center_input, assist_input,
                      out_dir='./utils/visualization',  # 特征图保存路径文件
                      save_feature=False,  # 是否以图片形式保存特征图
                      slice_number=5,
                      show_feature=True,  # 是否使用plt显示特征图
                      ):

    # feature = feature.detach().cpu()
    b, c, h, w = center_input.shape
    over_input = assist_input[:, :slice_number, :, :, :]
    under_input = assist_input[:, slice_number:, :, :, :]
    if b > 6:
        b = 6
    for i in range(b):
        figure = np.zeros(((h+30)*2, (w+30)*(slice_number+1)+30), dtype=np.uint8) + 255
        figure[10:h + 10, 10 + (w + 20) * 0: 10 + (w + 20) * 0 + w] = center_input[i, 0, :, :]*255
        for j in range(1, (slice_number+1)):
            overj = over_input[:, j-1, :, :, :]
            figure[10:h + 10, 10 + (w + 20) * j: 10 + (w + 20) * j + w] = overj[i, 0, :, :]*255
        for j in range(1, (slice_number+1)):
            underj = under_input[:, j-1, :, :, :]
            figure[30+h:30+h+h, 10 + (w + 20) * j: 10 + (w + 20) * j + w] = underj[i, 0, :, :]*255
        if save_feature:
            cv2.imwrite(out_dir + '/' + 'input' + '.png', figure)
        cv2.imshow("attention-" + str(c), figure)
        cv2.waitKey(0)

global layer
layer = 0

def attentionheatmap_visual(features,
                      out_dir='./Visualization/attention_af3/',  # 特征图保存路径文件
                      save_feature=True,  # 是否以图片形式保存特征图
                      show_feature=True,  # 是否使用plt显示特征图
                      feature_title=None,  # 特征图名字，默认以shape作为title
                      channel = None,
                      ):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    global layer
    b, c, h, w = features.shape
    if b > 1:
        b = 1
    for i in range(b):
        for j in range(c):
            featureij = features[i, j, :, :]
            featureij = featureij.cpu().detach().numpy()
            #fig = sns.heatmap(featureij, cmap="YlGnBu", vmin=0.0, vmax=0.005)  #Wistia, YlGnBu #0.005
            fig = sns.heatmap(featureij, cmap="coolwarm", vmin=-0.01, vmax=0.01) #-0.5,+0.5 , 0.003
            fig.set_xticks(range(0))
            #fig.set_xticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            fig.set_yticks(range(0))
            #fig.set_yticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            #sns.despine()
            plt.show()
            plt.close()
            fig_heatmap = fig.get_figure()
            imgpath = read_img_name()
            filename = os.path.basename(imgpath)
            filename = filename.split('.')[0]
            fig_heatmap.savefig(os.path.join(out_dir, filename + '_l' + str(layer) + '_' + str(j) + '.png'))
    layer = (layer + 1) % (12)


def attentionheatmap_visual3(features,
                      out_dir='./Visualization/attention_af3/',  # 特征图保存路径文件
                      save_feature=True,  # 是否以图片形式保存特征图
                      show_feature=True,  # 是否使用plt显示特征图
                      feature_title=None,  # 特征图名字，默认以shape作为title
                      channel = None,
                      ):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    global layer
    b, c, h, w = features.shape
    if b > 1:
        b = 1
    for i in range(b):
        for j in range(c):
            featureij = features[i, j, :, :]
            featureij = featureij.cpu().detach().numpy()
            #fig = sns.heatmap(featureij, cmap="YlGnBu", vmin=0.0, vmax=0.005)  #Wistia, YlGnBu #0.005
            fig = sns.heatmap(featureij, cmap="coolwarm", vmin=-0.01, vmax=0.01) #-0.5,+0.5 , 0.003
            fig.set_xticks(range(0))
            #fig.set_xticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            fig.set_yticks(range(0))
            #fig.set_yticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            #sns.despine()
            plt.show()
            plt.close()
            fig_heatmap = fig.get_figure()
            imgpath = read_img_name()
            filename = os.path.basename(imgpath)
            filename = filename.split('.')[0]
            fig_heatmap.savefig(os.path.join(out_dir, filename + '_l' + str(layer) + '_' + str(j) + '.png'))
    layer = (layer + 1) % (16)


def attentionheatmap_visual2(features, sita,
                      out_dir='./Visualization/attention_af3/',  # 特征图保存路径文件
                      value=0.05,
                      save_feature=True,  # 是否以图片形式保存特征图
                      show_feature=True,  # 是否使用plt显示特征图
                      feature_title=None,  # 特征图名字，默认以shape作为title
                      channel = None,
                      ):

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    global layer
    b, c, h, w = features.shape
    if b > 1:
        b = 1
    for i in range(b):
        for j in range(c):
            featureij = features[i, j, :, :]
            featureij = featureij.cpu().detach().numpy()
            #fig = sns.heatmap(featureij, cmap="YlGnBu", vmin=0.0, vmax=0.005)  #Wistia, YlGnBu #0.005
            fig = sns.heatmap(featureij, cmap="coolwarm", vmin=-value, vmax=value)  # 0.5 #0.003
            fig.set_xticks(range(0))
            #fig.set_xticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            fig.set_yticks(range(0))
            #fig.set_yticklabels(f'{c:.1f}' for c in np.arange(0.1, 1.01, 0.1))
            #sns.despine()
            plt.show()
            plt.close()
            fig_heatmap = fig.get_figure()
            imgpath = read_img_name()
            filename = os.path.basename(imgpath)
            filename = filename.split('.')[0]
            fig_heatmap.savefig(os.path.join(out_dir, filename + '_l' + str(layer) + '_' + str(j) + '_' + str(sita[j].item()) + '.png'))
    layer = (layer + 1) % (12)



def visual_segmentation(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        # img_r[seg0 == i] = table[i - 1, 0]
        # img_g[seg0 == i] = table[i - 1, 1]
        # img_b[seg0 == i] = table[i - 1, 2]
        img_r[seg0 == i] = table[i + 1 - 1, 0]
        img_g[seg0 == i] = table[i + 1 - 1, 1]
        img_b[seg0 == i] = table[i + 1 - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
    #img = cv2.addWeighted(img_ori0, 0.6, overlay, 0.4, 0) # ACDC
    img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0) # ISIC
    #img = np.uint8(0.3 * overlay + 0.7 * img_ori)
          
    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)


def visual_segmentation_binary(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = 255
        img_g[seg0 == i] = 255
        img_b[seg0 == i] = 255
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
          
    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, overlay)






def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed(backend='nccl')
    cfg = YAMLConfig(args.config, resume=args.resume, use_amp=args.amp)
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    model = cfg.model

    print(model)

    inputs = torch.randn(1,3,640,640)


    # # 前向传播,获取attention权重  
    # inputs = torch.randn(1, 512)
    # attention_weights = model.get_attention_map(inputs)

    # # 取第3层第2个head的attention map
    # attention_map = attention_weights[2][1] 

    # # 获取token个数  
    # num_tokens = attention_map.shape[1]

    # # 对attention map进行规范化处理
    # attention_map = attention_map.reshape(num_tokens, -1).mean(dim=1)
    # attention_map /= attention_map.max()
    # attention_map = attention_map.reshape(num_tokens, num_tokens)

    # # 绘制热力图
    # plt.figure(figsize=(6, 6))
    # plt.imshow(attention_map)
    # plt.colorbar()

    # # 设置坐标轴及标签
    # ticks = range(num_tokens)
    # plt.xticks(ticks, token_list[:num_tokens], rotation=90) 
    # plt.yticks(ticks, token_list[:num_tokens])

    # plt.show()




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='/root/autodl-tmp/code/RT-DETR/rtdetr_pytorch/configs/rtdetr/rtdetr_r18vd_6x_visdrone.yml', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)

    args = parser.parse_args()

    main(args)












