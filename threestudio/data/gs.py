import bisect
import random
from dataclasses import dataclass, field

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

import threestudio
from threestudio import register
from threestudio.utils.base import Updateable
from threestudio.utils.config import parse_structured
from mvdream.camera_utils import get_camera
from threestudio.utils.typing import *
import numpy as np

@dataclass
class GSLoadDataModuleConfig:
    # height, width, and batch_size should be Union[int, List[int]]
    # but OmegaConf does not support Union of containers
    source: str = "/home/project/gaussian-splatting/dataset/bear"
    height: Any = 512
    width: Any = 512
    batch_size: Any = 1
    resolution_milestones: List[int] = field(default_factory=lambda: [])
    eval_height: int = 512
    eval_width: int = 512
    eval_batch_size: int = 1
    n_val_views: int = 1
    n_test_views: int = 120
    elevation_range: Tuple[float, float] = (-10, 60)
    azimuth_range: Tuple[float, float] = (-180, 180)
    camera_distance_range: Tuple[float, float] = (4.,6.)
    fovy_range: Tuple[float, float] = (
        40,
        70,
    )  # in degrees, in vertical direction (along height)
    camera_perturb: float = 0.
    center_perturb: float = 0.
    up_perturb: float = 0.0
    light_position_perturb: float = 1.0
    light_distance_range: Tuple[float, float] = (0.8, 1.5)
    eval_elevation_deg: float = 15.0
    eval_camera_distance: float = 6.
    eval_fovy_deg: float = 70.0
    light_sample_strategy: str = "dreamfusion"
    batch_uniform_azimuth: bool = True
    progressive_until: int = 0  # progressive ranges for elevation, azimuth, r, fovy
    load_type: int = 0

class GSLoadDataset(Dataset):
    def __init__(self, cfg, split, scene) -> None:
        super().__init__()
        self.cfg: GSLoadDataModuleConfig = cfg
        self.split = split
        self.scene = scene
        self.total_view_num = len(self.scene.cameras)
        self.n_views = self.cfg.n_val_views

    def __len__(self):
        if self.split == "test": # 测试循环所有视图
            return self.total_view_num
        return self.n_views

    def __getitem__(self, index):
        # 随机抽取一个
        if self.split == "val": # 训练途中验证只需要一张图，随机抽取
            index = random.randint(0, self.total_view_num-1)
        return {
            "index": [index],
            "camera": [self.scene.cameras[index]]
        }
    
    def collate(self, batch):
        return batch
    
class GSLoadIterableDataset(IterableDataset):
    def __init__(self, cfg, split, scene) -> None:
        super().__init__()
        self.cfg: GSLoadDataModuleConfig = cfg
        self.split = split
        self.scene = scene
        self.total_view_num = len(self.scene.cameras)
        random.seed(0)  # make sure same views
        self.n2n_view_index = random.sample( # 创建一个随机的视图索引
            range(0, self.total_view_num),
            self.total_view_num
        )
        self.view_index_stack = self.n2n_view_index.copy()

    def collate(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        cam_list = []
        index_list = []
        for _ in range(self.cfg.batch_size):
            if not self.view_index_stack: # 视图索引为空就重新创建
                self.view_index_stack = self.n2n_view_index.copy()
            view_index = random.choice(self.view_index_stack)
            self.view_index_stack.remove(view_index)
            cam_list.append(self.scene.cameras[view_index])
            index_list.append(view_index)

        return {
            "index": index_list,
            "camera": cam_list
        }
    
    def collate_90(self, batch) -> Dict[str, Any]:
        # sample elevation angles
        cam_list = []
        index_list = []
        if not self.view_index_stack: # 视图索引为空就重新创建
            self.view_index_stack = self.n2n_view_index.copy()
        view_index_origin = random.choice(self.view_index_stack) # 选择一个初始视图
        view_index = []
        
        flag = 0
        elevation = 15 # 假设是15度
        if view_index_origin>=60: # 意味着是第二圈，每一圈的仰角都是相同的
            flag = 1
            elevation = 30 # 假设是30度
        index = view_index_origin%60 # 取模，到最后再根据flag还原
        # 根据index计算旋转角
        azimuth = ((index+60-15)%60)*6 # 初始角度

        camera = get_camera(4, elevation=elevation,
                azimuth_start=azimuth, azimuth_span=360)
        # print(camera.shape) # 4，16
        view_index.append(index+flag*60)
        # 依次将其旋转90、180、270度
        for i in range(self.cfg.batch_size-1):
            index = (index+15)%60
            view_index.append(index+flag*60)

        for i in range(self.cfg.batch_size):
            self.view_index_stack.remove(view_index[i])
            cam_list.append(self.scene.cameras[view_index[i]])
            index_list.append(view_index[i])
        # print(index_list)
        return {
            "index": index_list,
            "camera": cam_list,
            "c2w": camera
        }

    def __iter__(self):
        while True:
            yield {}

    def progressive_view(self, global_step):
        pass

@register("gs-load")
class GS_load(pl.LightningDataModule):
    cfg: GSLoadDataModuleConfig

    def __init__(self, cfg: Optional[Union[dict, DictConfig]] = None) -> None:
        from gaussiansplatting.scene.camera_scene import CamScene

        super().__init__()
        self.cfg = parse_structured(GSLoadDataModuleConfig, cfg)
        print(self.cfg.source,self.cfg.batch_size)
        self.train_scene = CamScene(self.cfg.source)
        self.eval_scene = CamScene(self.cfg.source)

    def setup(self, stage=None) -> None:
        if stage in [None, "fit"]:
            self.train_dataset = GSLoadIterableDataset(self.cfg, "train", self.train_scene) # 需要动态迭代
        if stage in [None, "fit", "validate"]:
            self.val_dataset = GSLoadDataset(self.cfg, "val", self.eval_scene)
        if stage in [None, "test", "predict"]:
            self.test_dataset = GSLoadDataset(self.cfg, "test", self.eval_scene)

    def prepare_data(self):
        pass

    def general_loader(self, dataset, batch_size, collate_fn=None) -> DataLoader:
        return DataLoader(
            dataset,
            num_workers=0,  # type: ignore
            batch_size=batch_size,
            collate_fn=collate_fn,
        )
    
    def train_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.train_dataset, batch_size=None, collate_fn=self.train_dataset.collate_90
        )
    
    def val_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.val_dataset,batch_size=None, collate_fn=self.val_dataset.collate
        )

    def test_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=None, collate_fn=self.test_dataset.collate
        )

    def predict_dataloader(self) -> DataLoader:
        return self.general_loader(
            self.test_dataset, batch_size=None, collate_fn=self.test_dataset.collate
        )

