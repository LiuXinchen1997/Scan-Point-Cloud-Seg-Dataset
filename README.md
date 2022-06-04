# Scan-Point-Cloud-Seg-Dataset

## 1 环境配置

`conda create -n <your_env_name> python=3.6 --yes`
`conda activate <your_env_name>`

`pip install -r requirements.txt`

## 2 处理流程

1. 使用MeshLab软件人工打标出前景点云
2. 使用generate_seg_label.py计算出背景点云、生成最终的点云标签文件
3. 使用generate_color_points.py生成带纹理的点云标签文件
4. 使用generate_train_test_split_data.py生成数据增强的训练数据与测试数据

## 3 效果展示

## 4 待添加

## 5 参考项目

[[1] Python Package: mesh-to-sdf](https://github.com/marian42/mesh_to_sdf)
