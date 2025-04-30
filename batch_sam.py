import os
import random
import argparse
import numpy as np
import torch
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
from dataclasses import dataclass, field
from typing import Tuple, Type
from copy import deepcopy
import torch
import torchvision
from torch import nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import maskclip
import torchvision.transforms as T

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

# 返回掩码提取的分割图像
def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8) # 背景是黑色的
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...] # 返回矩形区域的图像
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep
        
def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx
        
# *args 用于传递不定数量的非关键字参数，会将参数打包成一个元组(tuple)供函数内部使用，**kwargs 用于传递不定数量的关键字参数，会将参数打包成一个字典(dict)供函数内部使用
def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs) # 返回选中的mask索引
        masks_lvl = filter(keep_mask_nms, masks_lvl) # 过滤掉多余的mask

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image) # 生成掩码，这个是lang自己改写的SAM
    # pre-compute postprocess # 根据分数和掩码之间的重叠率删除多余的掩码
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    def mask2segmap(masks, image):
        seg_img_list = []
        seg_map = -np.ones(image.shape[:2], dtype=np.int32) # 生成空的标记图
        for i in range(len(masks)):
            mask = masks[i]
            seg_img = get_seg_img(mask, image) # 返回掩码提取的分割图像
            pad_seg_img = cv2.resize(pad_img(seg_img), (224,224))
            seg_img_list.append(pad_seg_img)

            seg_map[masks[i]['segmentation']] = i # 标记掩码的位置
        seg_imgs = np.stack(seg_img_list, axis=0) # b,H,W,3
        seg_imgs = (torch.from_numpy(seg_imgs.astype("float32")).permute(0,3,1,2) / 255.0).to('cuda')

        return seg_imgs, seg_map

    seg_images, seg_maps = {}, {}
    seg_images['default'], seg_maps['default'] = mask2segmap(masks_default, image)
    if len(masks_s) != 0:
        seg_images['s'], seg_maps['s'] = mask2segmap(masks_s, image)
    if len(masks_m) != 0:
        seg_images['m'], seg_maps['m'] = mask2segmap(masks_m, image)
    if len(masks_l) != 0:
        seg_images['l'], seg_maps['l'] = mask2segmap(masks_l, image)
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

class TorchPCA(object):

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self

    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected

def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    device = image_feats_list[0].device

    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()

    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)

    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)

    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))

    return reduced_feats, fit_pca

class PCAUnprojector(nn.Module):

    def __init__(self, feats, dim, device, use_torch_pca=False, **kwargs):
        super().__init__()
        self.dim = dim

        if feats is not None:
            self.original_dim = feats.shape[1]
        else:
            self.original_dim = kwargs["original_dim"]

        if self.dim != self.original_dim:
            if feats is not None:
                sklearn_pca = pca([feats], dim=dim, use_torch_pca=use_torch_pca)[1]

                # Register tensors as buffers
                self.register_buffer('components_',
                                     torch.tensor(sklearn_pca.components_, device=device, dtype=feats.dtype))
                self.register_buffer('singular_values_',
                                     torch.tensor(sklearn_pca.singular_values_, device=device, dtype=feats.dtype))
                self.register_buffer('mean_', torch.tensor(sklearn_pca.mean_, device=device, dtype=feats.dtype))
            else:
                self.register_buffer('components_', kwargs["components_"].t())
                self.register_buffer('singular_values_', kwargs["singular_values_"])
                self.register_buffer('mean_', kwargs["mean_"])

        else:
            print("PCAUnprojector will not transform data")

    def forward(self, red_feats):

        if self.dim == self.original_dim:
            return red_feats
        else:
            b, c, h, w = red_feats.shape
            red_feats_reshaped = red_feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
            unprojected = (red_feats_reshaped @ self.components_) + self.mean_.unsqueeze(0)
            return unprojected.reshape(b, h, w, self.original_dim).permute(0, 3, 1, 2)

    def project(self, feats):

        if self.dim == self.original_dim:
            return feats
        else:
            b, c, h, w = feats.shape
            feats_reshaped = feats.permute(0, 2, 3, 1).reshape(b * h * w, c)
            t0 = feats_reshaped - self.mean_.unsqueeze(0).to(feats.device)
            projected = t0 @ self.components_.t().to(feats.device)
            return projected.reshape(b, h, w, self.dim).permute(0, 3, 1, 2)
        
# 加载上采样之后的特征数据
class Autoencoder_dataset(Dataset):
    def __init__(self, data_dir, image_dir):
        
        self.name_list = [f for f in os.listdir(data_dir) if f.endswith('.pth')] # 上采样器保存文件名
        self.name_list.sort()
#         print(self.name_list)
        self.path_list = [os.path.join(data_dir, f) for f in self.name_list] # 上采样器保存路径
        self.image_path_list = [os.path.join(image_dir, f.replace('pth', 'jpg')) for f in self.name_list]
        self.transform = T.Compose([
            T.Resize((h, w)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
    def __getitem__(self, index):
        with torch.no_grad():
            up_path = self.path_list[index] # 获取上采样器路径
            name = self.name_list[index]
            checkpoint = torch.load(up_path)
            # 从checkpoint中获取模型和unprojector
            upsampler = checkpoint["model"].eval()
            unprojector = checkpoint["unprojector"]
            image_path = self.image_path_list[index]
            original_image = self.transform(Image.open(image_path).convert('RGB')).unsqueeze(0).cuda()
            hr_feats = upsampler(original_image) # 128维
            hr_feats = unprojector(hr_feats) # 512维 [1, 512, 365, 494]
            hr_feats = hr_feats.permute(0,2,3,1)
            hr_feats /= hr_feats.norm(dim=-1, keepdim=True) # 归一化
            return hr_feats,name

    def __len__(self):
        return len(self.name_list)
    
if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num) # 设置所有的随机种子

    sam_ckpt_path = "ckpts/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')

    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )
    dataset_path = "/home/project/gaussian-splatting/dataset/figurines"
    img_folder = os.path.join(dataset_path, 'images') # 图像文件夹路径
    data_list = os.listdir(img_folder) # 图像文件名列表
    data_list.sort()
    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        
        if orig_h > 1080:
            if not WARNED:
                print("[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                    "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                WARNED = True
            global_down = orig_h / 1080
        else:
            global_down = 1
        

        scale = float(global_down)
        resolution = (int( orig_w  / scale), int(orig_h / scale))

        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)


    # maskclip数据加载
    up_path = "/home/project/MaskCLIP/notebooks/up/figurines" # 上采样器保存路径
    
    scale = 2
    h = 728//scale
    w = 986//scale
    device="cuda"
    
    train_dataset = Autoencoder_dataset(up_path,img_folder) # 数据集加载
    train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False
        )
    model_clip, preprocess = maskclip.load("/home/project/MaskCLIP/model/ViT-B-16.pt", input_resolution=(h,w), device="cuda")
    model_clip = model_clip.eval()
    for idx, feature in tqdm(enumerate(train_loader)):
        image = imgs[idx].unsqueeze(0)
        _, seg_map = sam_encoder(image)
        image_feature = nn.functional.interpolate(feature[0][0].permute(0,3,1,2), size=(728,986), mode='bilinear', align_corners=False) # [1, 512, 1080, 1440]
        image_feature = image_feature.permute(2,3,1,0).squeeze()
        seg_mask = torch.from_numpy(seg_map['s'])
        avg_list = []
        for i in (range(seg_mask.min(),seg_mask.max()+1)):
            indices = np.where(seg_mask == i)
            
            all_i_feature = image_feature[indices] # 第i个mask的所有特征
            all_i_feature /= all_i_feature.norm(dim=-1, keepdim=True)
            mean_i_feature = torch.mean(all_i_feature, dim=0)
            mean_i_feature /= mean_i_feature.norm(dim=-1, keepdim=True) # 归一化
            avg_list.append(mean_i_feature.unsqueeze(0))
        avg_list = torch.cat(avg_list, dim=0)
        
        # torch.save(seg_mask.cpu(), os.path.join(dataset_path,"avg_feature", "frame_{:05d}_s.pt".format(idx)))
        # torch.save(avg_list.cpu(), os.path.join(dataset_path,"avg_feature", "frame_{:05d}_a.pt".format(idx)))
        torch.save(seg_mask.cpu(), os.path.join(dataset_path,"avg_feature", feature[1][0].replace('.pth', '_s.pt')))
        torch.save(avg_list.cpu(), os.path.join(dataset_path,"avg_feature", feature[1][0].replace('.pth', '_a.pt')))
            