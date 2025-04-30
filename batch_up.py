from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from kornia.filters import gaussian_blur2d
from torch.utils.data import DataLoader
from tqdm import tqdm
from featup.datasets.JitteredImage import JitteredImage, apply_jitter
from featup.datasets.util import get_dataset, SlicedDataset
from featup.layers import ImplicitFeaturizer, MinMaxScaler, ChannelNorm
from featup.losses import total_variation
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import Dataset
import numpy as np
import torchvision
from torch import nn
from dataclasses import dataclass, field
from typing import Tuple, Type
import requests
from pkg_resources import packaging
import maskclip
from featup.util import (norm as reg_norm, unnorm as reg_unorm, generate_subset,
                         midas_norm, midas_unnorm, pca, PCAUnprojector, prep_image)
import os

def mag(t):
    return t.square().sum(1, keepdim=True).sqrt()

# 上采样网络
def get_implicit_upsampler(start_dim, end_dim, color_feats, n_freqs):
    return torch.nn.Sequential(
        MinMaxScaler(),
        ImplicitFeaturizer(color_feats, n_freqs=n_freqs, learn_bias=True),
        ChannelNorm(start_dim),
        torch.nn.Dropout2d(p=.2),
        torch.nn.Conv2d(start_dim, end_dim, 1),
        torch.nn.ReLU(),
        torch.nn.Dropout2d(p=.2),
        ChannelNorm(end_dim),
        torch.nn.Conv2d(end_dim, end_dim, 1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(end_dim, end_dim, 1),
    )


# 下采样网络
class AttentionDownsampler(torch.nn.Module):

    def __init__(self, dim, kernel_size, final_size, blur_attn, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.final_size = final_size
        self.stride = stride
        self.in_dim = dim
        self.attention_net = torch.nn.Sequential(
            torch.nn.Dropout(p=.2),
            torch.nn.Linear(self.in_dim, 1)
        )
        self.w = torch.nn.Parameter(torch.ones(kernel_size, kernel_size).cuda()
                                    + .01 * torch.randn(kernel_size, kernel_size).cuda())
        self.b = torch.nn.Parameter(torch.zeros(kernel_size, kernel_size).cuda()
                                    + .01 * torch.randn(kernel_size, kernel_size).cuda())
        self.blur_attn = blur_attn

    def forward_attention(self, feats, guidance):
        return self.attention_net(feats.permute(0, 2, 3, 1)).squeeze(-1).unsqueeze(1)

    def forward(self, hr_feats, guidance):
        b, c, h, w = hr_feats.shape

        if self.blur_attn:
            inputs = gaussian_blur2d(hr_feats, 5, (1.0, 1.0))
        else:
            inputs = hr_feats


        patches = torch.nn.Unfold(self.kernel_size, stride=self.stride)(inputs) \
            .reshape(
            (b, self.in_dim, self.kernel_size * self.kernel_size, self.final_size[0], self.final_size[1])) \
            .permute(0, 3, 4, 2, 1)

        patch_logits = self.attention_net(patches).squeeze(-1)

        b, h, w, p = patch_logits.shape
        dropout = torch.rand(b, h, w, 1, device=patch_logits.device) > 0.2

        w = self.w.flatten().reshape(1, 1, 1, -1)
        b = self.b.flatten().reshape(1, 1, 1, -1)

        patch_attn_logits = (patch_logits * dropout) * w + b
        patch_attention = F.softmax(patch_attn_logits, dim=-1)

        downsampled = torch.einsum("bhwpc,bhwp->bchw", patches, patch_attention)

        return downsampled[:, :c, :, :]
    
class SampleImage(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        batch = {
            "img": image,
            "img_path": image_path
        }
        return batch

    def __len__(self):
        return len(self.paths)

if __name__ == "__main__":

    # 超参数设置
    scale = 2
    device = "cuda"
    h = 738//scale
    w = 994//scale
    kernel_size = 16
    stride = kernel_size
    n_images = 3000 # 产生多少个变换后的图像
    use_flips = True
    max_zoom = 1.8
    max_pad = 30
    featurize_batch_size = 16 # batch_size，每次产生featurize_batch_size个低分辨率特征
    pca_batch = 50 # 用于计算pca的特征个数
    proj_dim = 128 # pca后的特征维度
    n_freqs = 30
    blur_attn = True
    step = 1200 # 反向传播多少次
    inner_batch = 10 # 每次用于计算损失的低分辨率特征个数
    mag_tv_weight = 0.05
    mag_weight = 0.001
    blur_pin = 0.1

    # 加载maskclip模型
    model, preprocess = maskclip.load("/home/project/MaskCLIP/model/ViT-B-16.pt", input_resolution=(h,w), device=device)
    model = model.eval()
    channelnorm = ChannelNorm(512).to(device)

    # 图像处理
    transform = T.Compose([
        T.Resize((h, w)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    img_path = "/home/project/gaussian-splatting/dataset/teaser" # 图像文件夹
    img_name_list = [f for f in os.listdir(img_path) if f.endswith('.jpg')] # 图像名称列表
    img_path_list = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith('.jpg')] # 图像绝对路径列表

    full_dataset = SampleImage(
        paths=img_path_list,
        transform=transform
    )


    dataset = full_dataset
    loader = DataLoader(dataset, shuffle=False)
    for img_num, batch in enumerate(loader):
        original_image = batch["img"].cuda()

        # 当前正在处理的图像
        print(f"up/teaser/{batch['img_path'][0][batch['img_path'][0].rfind('/') + 1:].replace('.jpg', '')}.pth")

        dataset = JitteredImage(original_image, n_images, use_flips, max_zoom, max_pad)
        loader = DataLoader(dataset, featurize_batch_size)
        with torch.no_grad():
            transform_params = defaultdict(list) # 保存每张图像的变换参数
            _,lr_feats = model.encode_image(original_image)
            lr_feats = channelnorm(lr_feats.permute(0,2,1).reshape(-1,512,h//kernel_size,w//kernel_size).detach().to(torch.float32))
            [red_lr_feats], fit_pca = pca([lr_feats], dim=9, use_torch_pca=True)
            jit_features = []
            for transformed_image, tp in (loader):
                for k, v in tp.items():
                    transform_params[k].append(v) # 存储变换参数
                _,transformed_lr_feats = model.encode_image(transformed_image)
                transformed_lr_feats = channelnorm(transformed_lr_feats.permute(0,2,1).reshape(-1,512,h//kernel_size,w//kernel_size).detach().to(torch.float32))
                jit_features.append(transformed_lr_feats.cpu())
            jit_features = torch.cat(jit_features, dim=0)
            transform_params = {k: torch.cat(v, dim=0) for k, v in transform_params.items()}

            unprojector = PCAUnprojector(jit_features[:pca_batch], proj_dim, lr_feats.device,
                                        use_torch_pca=True)
            jit_features = unprojector.project(jit_features) # pca变换，降维 变换后低分辨率特征
            lr_feats = unprojector.project(lr_feats)
            torch.cuda.empty_cache()


        # 模型初始化
        params = []
        end_dim = proj_dim
        start_dim = 5 * n_freqs * 2 + 3
        upsampler = get_implicit_upsampler(start_dim, end_dim, True, n_freqs).cuda()
        params.append({"params": upsampler.parameters()})
        final_size = (h//kernel_size,w//kernel_size)
        downsampler = AttentionDownsampler(proj_dim + 1, kernel_size, final_size, blur_attn, stride).cuda()
        params.append({"params": downsampler.parameters()})
        with torch.no_grad():
            outlier_detector = torch.nn.Conv2d(proj_dim, 1, 1).cuda()
            outlier_detector.weight.copy_(outlier_detector.weight * .1)
            outlier_detector.bias.copy_(outlier_detector.bias * .1)

        params.append({"params": outlier_detector.parameters()})
        get_scale = lambda feats: torch.exp(outlier_detector(feats) + .1).clamp_min(.0001)

        # 训练模型
        optim = torch.optim.NAdam(params)
        for i in tqdm(range(step)):
            upsampler.train()
            downsampler.train()

            hr_feats = upsampler(original_image)
            hr_mag = mag(hr_feats)
            hr_both = torch.cat([hr_mag, hr_feats], dim=1)
            loss = 0.0

            target = []
            hr_feats_transformed = []
            for j in range(inner_batch):
                idx = torch.randint(n_images, size=())
                target.append(jit_features[idx].unsqueeze(0))
                selected_tp = {k: v[idx] for k, v in transform_params.items()}
                hr_feats_transformed.append(apply_jitter(hr_both, max_pad, selected_tp))

            target = torch.cat(target, dim=0).cuda(non_blocking=True)
            hr_feats_transformed = torch.cat(hr_feats_transformed, dim=0) # [10, 129, 182, 247]

            output_both = downsampler(hr_feats_transformed, None)
            magnitude = output_both[:, 0:1, :, :]
            output = output_both[:, 1:, :, :] # [10, 128, 11, 15]
            scales = get_scale(target)
            rec_loss = ((1 / (2 * scales ** 2)) * (output - target).square() + scales.log()).mean()

            loss += rec_loss
            
            if mag_weight > 0.0:
                mag_loss = (magnitude - mag(target)).square().mean()
                mag_loss2 = (mag(output) - mag(target)).square().mean()
                loss += mag_loss * mag_weight

            if mag_tv_weight > 0.0:
                mag_tv = total_variation(hr_mag)
                loss += mag_tv_weight * mag_tv

            if blur_pin > 0.0:
                blur_pin_loss = (gaussian_blur2d(hr_feats, 5, (1.0, 1.0)) - hr_feats).square().mean()
                loss += blur_pin * blur_pin_loss
            
            loss.backward()
            

            optim.step()
            optim.zero_grad()

        # 保存训练好的模型
        torch.save({"model": upsampler, "unprojector": unprojector}, f"up/teaser/{batch['img_path'][0][batch['img_path'][0].rfind('/') + 1:].replace('.jpg', '')}.pth")




