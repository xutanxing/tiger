from dataclasses import dataclass, field
import torch
import threestudio
from threestudio.systems.base import BaseLift3DSystem
from threestudio.utils.ops import binary_cross_entropy, dot
from threestudio.utils.typing import *
from gaussiansplatting.gaussian_renderer import render
from gaussiansplatting.scene import Scene, GaussianModel
from gaussiansplatting.arguments import ModelParams, PipelineParams, get_combined_args,OptimizationParams
from gaussiansplatting.scene.cameras import Camera
from argparse import ArgumentParser, Namespace
import os
from pathlib import Path
from plyfile import PlyData, PlyElement
from gaussiansplatting.utils.sh_utils import SH2RGB
from gaussiansplatting.scene.gaussian_model import BasicPointCloud
import numpy as np
# from shap_e.diffusion.sample import sample_latents
# from shap_e.diffusion.gaussian_diffusion import diffusion_from_config as diffusion_from_config_shape
# from shap_e.models.download import load_model, load_config
# from shap_e.util.notebooks import create_pan_cameras, decode_latent_images, gif_widget
# from shap_e.util.notebooks import decode_latent_mesh
import io  
from PIL import Image  
import open3d as o3d


@threestudio.register("two-system")
class GaussianDreamer(BaseLift3DSystem):
    @dataclass
    class Config(BaseLift3DSystem.Config):
        radius: float = 4
        sh_degree: int = 0
        load_type: int = 0
        load_path: str = "./load/shapes/stand.obj"

    cfg: Config
    def configure(self) -> None:
        self.radius = self.cfg.radius
        self.sh_degree =self.cfg.sh_degree
        self.load_type =self.cfg.load_type
        self.load_path = self.cfg.load_path

        self.gaussian = GaussianModel(sh_degree = self.sh_degree)
        bg_color = [1, 1, 1] if False else [0, 0, 0] # 背景必定为黑色
        self.background_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self.step = 0
    def forward(self, batch: Dict[str, Any],renderbackground = None) -> Dict[str, Any]:
        # print(len(batch["camera"]))
        # print(batch["camera"][0])
        if renderbackground is None:
            renderbackground = self.background_tensor
        images = []
        depths = []
        self.viewspace_point_list = []
        for id in range(len(batch["camera"])):
            
            # viewpoint_cam = Camera(c2w = batch['c2w_3dgs'][id],FoVy = batch['fovy'][id],height = batch['height'],width = batch['width'])
            viewpoint_cam = batch["camera"][id]

            render_pkg = render(viewpoint_cam, self.gaussian, self.pipe, renderbackground)
            image, viewspace_point_tensor, _, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            self.viewspace_point_list.append(viewspace_point_tensor)

            
            if id == 0:

                self.radii = radii
            else:


                self.radii = torch.max(radii,self.radii)
                
            
            depth = render_pkg["depth_3dgs"]
            depth =  depth.permute(1, 2, 0)
            
            image =  image.permute(1, 2, 0)
            images.append(image)
            depths.append(depth)
            



        images = torch.stack(images, 0)
        depths = torch.stack(depths, 0)
        self.visibility_filter = self.radii>0.0
        render_pkg["comp_rgb"] = images
        render_pkg["depth"] = depths
        render_pkg["opacity"] = depths / (depths.max() + 1e-5)
        return {
            **render_pkg,
        }

    def on_fit_start(self) -> None:
        super().on_fit_start()
        # only used in training
        self.prompt_processor = threestudio.find(self.cfg.prompt_processor_type)(
            self.cfg.prompt_processor
        )
        self.guidance = threestudio.find(self.cfg.guidance_type)(self.cfg.guidance)

        self.second_prompt_processor = threestudio.find(self.cfg.prompt_processor_type)( # mvprompt编码器
            self.cfg.second_prompt_processor
        )
        self.second_guidance = threestudio.find(self.cfg.second_guidance_type)(self.cfg.second_guidance) # mvguidance
        self.second_guidance.requires_grad_(False)

    def training_step(self, batch, batch_idx):

        self.gaussian.update_learning_rate(self.true_global_step)
        
        # if self.true_global_step > 500: # 不要更新，一开始已经很小了
        #     self.guidance.set_min_max_steps(min_step_percent=0.02, max_step_percent=0.1)

        self.gaussian.update_learning_rate(self.true_global_step)

        out = self(batch)

        # 其实这两步可以提到on_fit_start里面
        prompt_utils = self.prompt_processor()
        second_prompt_utils = self.second_prompt_processor()

        images = out["comp_rgb"]
        # print(prompt_utils.use_perp_neg) # false

        
        
        tmp = torch.zeros(images.shape[0])
        cond_images = torch.concatenate([batch["camera"][idx].original_image.unsqueeze(0).permute(0,2,3,1) for idx in range(images.shape[0])], dim=0)
        # print(cond_images.shape) # [4, 738, 994, 3]
        # print(images.shape) # [4, 738, 994, 3]

        # guidance_out = self.guidance(
        #     images, prompt_utils, tmp,tmp,tmp, rgb_as_latents=False,guidance_eval=guidance_eval
        # )
        # ---------------------------新增----------------------------
        
        guidance_out = self.guidance(images, cond_images, prompt_utils)
        
        second_guidance_out = self.second_guidance(images, second_prompt_utils, batch["c2w"])

        loss = 0.0
        p2p_weight = float(self.step)/2000.0 # 随着epoch增加，p2p_weight逐渐增大
        if self.step>2000:
            p2p_weight = 1.0
            
        self.step+=1

        print("instructp2p_loss: ",guidance_out['loss_sds']," mv_loss: ",second_guidance_out['loss_sds']," p2p_weight: ",p2p_weight) # 对比loss数量级
        # (不加camera)p2p_loss范围在50左右，大部分小于50，mv_loss范围在数百到数千，大部分在两千左右
        # (加camera)，mv_loss范围仍在几百到数千，不过大部分在1000左右
        
        loss = loss + 0.5*(1+p2p_weight)*guidance_out['loss_sds'] *self.C(self.cfg.loss['lambda_sds'])
        loss = loss + 0.2*(1-p2p_weight)*second_guidance_out['loss_sds'] *self.C(self.cfg.loss['lambda_sds'])
        
        loss_sparsity = (out["opacity"] ** 2 + 0.01).sqrt().mean()
        self.log("train/loss_sparsity", loss_sparsity)
        loss += loss_sparsity * self.C(self.cfg.loss.lambda_sparsity)

        opacity_clamped = out["opacity"].clamp(1.0e-3, 1.0 - 1.0e-3)
        loss_opaque = binary_cross_entropy(opacity_clamped, opacity_clamped)
        self.log("train/loss_opaque", loss_opaque)
        loss += loss_opaque * self.C(self.cfg.loss.lambda_opaque)
        
        for name, value in self.cfg.loss.items():
            self.log(f"train_params/{name}", self.C(value))

        return {"loss": loss}



    def on_before_optimizer_step(self, optimizer):
        # print(self.gaussian._xyz.shape) # 打印Gaussian点的个数（一开始是300多万），从400代开始发生变化，500代的时候激增到700多万
        max_densify_percent = 0.01 # 最多只能增加1%的密度，而且是有grad的数量的1%，避免显存溢出
        with torch.no_grad():
            
            if self.true_global_step < 2000: # 15000
                viewspace_point_tensor_grad = torch.zeros_like(self.viewspace_point_list[0])
                for idx in range(len(self.viewspace_point_list)):
                    viewspace_point_tensor_grad = viewspace_point_tensor_grad + self.viewspace_point_list[idx].grad
                # Keep track of max radii in image-space for pruning
                self.gaussian.max_radii2D[self.visibility_filter] = torch.max(self.gaussian.max_radii2D[self.visibility_filter], self.radii[self.visibility_filter])
                
                self.gaussian.add_densification_stats(viewspace_point_tensor_grad, self.visibility_filter)

                if self.true_global_step > 300 and self.true_global_step % 100 == 0: # 500 100
                    size_threshold = 20 if self.true_global_step > 500 else None # 3000
                    self.gaussian.densify_and_prune(0.0002 , 0.05, self.cameras_extent, size_threshold, max_densify_percent) # 不透明度小于0.05就删掉

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        self.save_image_grid(
            f"it{self.true_global_step}-{batch['index'][0]}.png",
            (
                [
                    {
                        "type": "rgb",
                        "img": batch["rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    }
                ]
                if "rgb" in batch
                else []
            )
            + [
                {
                    "type": "rgb",
                    "img": out["comp_rgb"][0],
                    "kwargs": {"data_format": "HWC"},
                },
            ]
            + (
                [
                    {
                        "type": "rgb",
                        "img": out["comp_normal"][0],
                        "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                    }
                ]
                if "comp_normal" in out
                else []
            ),
            name="validation_step",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"it{self.true_global_step}-val.ply")
        # self.gaussian.save_ply(save_path)
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-val-color.ply"))

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        only_rgb = True
        bg_color = [1, 1, 1] if False else [0, 0, 0] # 背景颜色始终是黑色

        testbackground_tensor = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        out = self(batch,testbackground_tensor)
        if only_rgb:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                ),
                name="test_step",
                step=self.true_global_step,
            )
        else:
            self.save_image_grid(
                f"it{self.true_global_step}-test/{batch['index'][0]}.png",
                (
                    [
                        {
                            "type": "rgb",
                            "img": batch["rgb"][0],
                            "kwargs": {"data_format": "HWC"},
                        }
                    ]
                    if "rgb" in batch
                    else []
                )
                + [
                    {
                        "type": "rgb",
                        "img": out["comp_rgb"][0],
                        "kwargs": {"data_format": "HWC"},
                    },
                ]
                + (
                    [
                        {
                            "type": "rgb",
                            "img": out["comp_normal"][0],
                            "kwargs": {"data_format": "HWC", "data_range": (0, 1)},
                        }
                    ]
                    if "comp_normal" in out
                    else []
                )
                + (
                    [
                        {
                            "type": "grayscale",
                            "img": out["depth"][0],
                            "kwargs": {},
                        }
                    ]
                    if "depth" in out
                    else []
                )
                + [
                    {
                        "type": "grayscale",
                        "img": out["opacity"][0, :, :, 0],
                        "kwargs": {"cmap": None, "data_range": (0, 1)},
                    },
                ],
                name="test_step",
                step=self.true_global_step,
            )


    def on_test_epoch_end(self):
        self.save_img_sequence(
            f"it{self.true_global_step}-test",
            f"it{self.true_global_step}-test",
            "(\d+)\.png",
            save_format="mp4",
            fps=10,
            name="test",
            step=self.true_global_step,
        )
        # save_path = self.get_save_path(f"last_3dgs.ply")
        # self.gaussian.save_ply(save_path)
        
        # if self.load_type==0:
        #     o3d.io.write_point_cloud(self.get_save_path("shape.ply"), self.point_cloud)
        #     self.save_gif_to_file(self.shapeimages, self.get_save_path("shape.gif"))
        # load_ply(save_path,self.get_save_path(f"it{self.true_global_step}-test-color.ply"))
        


    def configure_optimizers(self):
        self.parser = ArgumentParser(description="Training script parameters")
        
        opt = OptimizationParams(self.parser)
        # point_cloud = self.pcb()
        self.cameras_extent = 5.567721414566041 # 数据集读取之后可以获得
        # self.gaussian.create_from_pcd(point_cloud, self.cameras_extent)
        # 直接使用已经有的Gaussian初始化Gaussian
        checkpoint = "/home/project/gaussian-splatting/output/bear_avg/chkpntbear_Statue.pth"
        (model_params, first_iter) = torch.load(checkpoint)
        self.gaussian.restore(model_params, opt)
        self.pipe = PipelineParams(self.parser)
        self.gaussian.training_setup(opt)
        
        ret = {
            "optimizer": self.gaussian.optimizer,
        }

        return ret
