# 新增文件
import os
from gaussiansplatting.scene.dataset_readers import sceneLoadTypeCallbacks
from gaussiansplatting.utils.camera_utils import cameraList_load


class CamScene:
    def __init__(self, source_path): # 这里输入的h、w没有意义
        
        scene_info = sceneLoadTypeCallbacks["Colmap"](source_path, None, False)
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        self.cameras = cameraList_load(scene_info.train_cameras)
        # 打印出来，为了给gaussiandreamer那边赋值cameras_extent
        print("cameras_extent:", self.cameras_extent)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]