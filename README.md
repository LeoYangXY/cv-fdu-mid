训练模型 mask-rcnn：
python tools/train.py configs/mask_rcnn/mask-rcnn_r50_fpn_1x_voc.py 

训练模型 sparse-rcnn：
python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_voc.py 

测试模型 mask-rcnn:
python demo/task_maskrcnn.py
便能够由data文件夹下的原始图像生成目标检测+实例分割的图像

测试模型 sparse-rcnn：
python demo/task_sparsercnn.py
便能够由data文件夹下的原始图像生成目标检测+实例分割的图像
