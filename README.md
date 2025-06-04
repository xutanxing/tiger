![block](./images/teaser.png)
# TIGER: Text-Instructed 3D Gaussian Retrieval and Coherent Editing
Thanks to GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models for the source code. This repository is implemented based on it.

What this repository provides now is editing related code.

## The places that need to be modified are as follows：

- threestudio\data\gs.py source(data set)
- threestudio\systems\two.py checkpoint(source scene)
- gaussiansplatting\scene\dataset_readers.py train_cam_infos(camera pose)

**Quickstart**
```
python launch.py --config configs/gaussiandreamer-two.yaml --train --gpu 0
```
