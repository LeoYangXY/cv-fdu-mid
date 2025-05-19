# mask_rcnn_voc_config.py
_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',  # 使用 Mask R-CNN 的 base 模型
    '../_base_/datasets/voc0712.py', 
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_1x.py'
]
