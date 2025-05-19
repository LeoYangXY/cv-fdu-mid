import mmcv
import torch
import os
import time
import numpy as np
from mmdet.apis import init_detector, inference_detector
from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample
from mmdet.registry import VISUALIZERS

# ------------------------------
# 配置部分
# ------------------------------
config_path = 'configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py'
checkpoint_path = 'work_dirs/mask-rcnn_r50_fpn_1x_voc/good_chekpoint.pth'
input_dir = 'data/new_pic'
output_dir = 'data/mask_rcnn/result'
os.makedirs(output_dir, exist_ok=True)
score_threshold = 0.3
max_image_size = 1600

# COCO 80类别
COCO_CLASSES = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
)

# VOC 20类别（完整列表）
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# COCO到VOC的类别映射（完整映射）
COCO_TO_VOC = {
    # 格式: COCO索引 : VOC索引
    # 交通工具类
    0: 14,   # person -> person
    1: 1,    # bicycle -> bicycle
    2: 6,    # car -> car
    3: 13,   # motorcycle -> motorbike
    4: 0,    # airplane -> aeroplane
    5: 5,    # bus -> bus
    6: 18,   # train -> train
    7: 6,    # truck -> car (VOC无truck)
    8: 3,    # boat -> boat
    # 动物类
    14: 2,   # bird -> bird
    15: 7,   # cat -> cat
    16: 11,  # dog -> dog
    17: 12,  # horse -> horse
    18: 16,  # sheep -> sheep
    19: 9,   # cow -> cow
    # 家具类
    56: 8,   # chair -> chair
    57: 17,  # couch -> sofa
    59: 15,  # potted plant -> pottedplant
    60: 10,  # dining table -> diningtable
    62: 19,  # tv -> tvmonitor
    # 其他
    9: 4,    # traffic light -> bottle (无对应)
    43: 4,   # bottle -> bottle
}
# ------------------------------
# 模型初始化
# ------------------------------
def init_model():
    print(f"[{time.strftime('%H:%M:%S')}] ⚙️ 初始化模型...")
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = init_detector(config_path, checkpoint_path, device=device)
    model.dataset_meta = {'classes': COCO_CLASSES}
    return model

# ------------------------------
# 结果处理
# ------------------------------
def process_results(results, img_shape):
    h, w = img_shape[:2]
    pred = results.pred_instances
    
    # 坐标修正
    bboxes = pred.bboxes.clone()
    bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], 0, h)
    
    # 类别映射
    valid_mask = torch.zeros_like(pred.labels, dtype=torch.bool)
    converted_labels = torch.zeros_like(pred.labels)
    
    for i, label in enumerate(pred.labels):
        coco_idx = int(label)
        if coco_idx in COCO_TO_VOC:
            valid_mask[i] = True
            converted_labels[i] = COCO_TO_VOC[coco_idx]
    
    # 综合过滤
    keep_mask = (pred.scores > score_threshold) & valid_mask
    
    filtered = DetDataSample()
    filtered.pred_instances = InstanceData(
        bboxes=bboxes[keep_mask],
        scores=pred.scores[keep_mask],
        labels=converted_labels[keep_mask],
        masks=pred.masks[keep_mask] if hasattr(pred, 'masks') else None
    )
    return filtered
# ------------------------------
# 可视化工具（关键修改）
# ------------------------------
class ResultVisualizer:
    def __init__(self, model):
        self.visualizer = VISUALIZERS.build(model.cfg.visualizer)
        self.visualizer.dataset_meta = {'classes': VOC_CLASSES}
    
    def draw_detection(self, img, result, out_path):
        """仅绘制检测框（关键修改：确保不显示掩膜）"""
        # 创建不包含掩膜的临时结果
        det_result = DetDataSample()
        det_result.pred_instances = InstanceData(
            bboxes=result.pred_instances.bboxes,
            scores=result.pred_instances.scores,
            labels=result.pred_instances.labels
        )
        
        # 新版本MMDetection使用draw_instances
        if hasattr(self.visualizer, 'draw_instances'):
            self.visualizer.draw_instances(
                image=img,
                instances=det_result.pred_instances,
                draw_heatmap=False,
                show=False,
                out_file=out_path
            )
        # 旧版本兼容
        else:
            self.visualizer.add_datasample(
                'det',
                img,
                data_sample=det_result,
                draw_gt=False,
                show=False,
                out_file=out_path
            )
    
    def draw_segmentation(self, img, result, out_path):
        """绘制检测框和掩膜"""
        if not hasattr(result.pred_instances, 'masks'):
            print("警告：当前结果不包含掩膜信息")
            return
            
        # 新版本MMDetection使用draw_instances
        if hasattr(self.visualizer, 'draw_instances'):
            self.visualizer.draw_instances(
                image=img,
                instances=result.pred_instances,
                draw_heatmap=False,
                show=False,
                out_file=out_path
            )
        # 旧版本兼容
        else:
            self.visualizer.add_datasample(
                'seg',
                img,
                data_sample=result,
                draw_gt=False,
                show=False,
                out_file=out_path
            )

# ------------------------------
# 保存结果（关键修改）
# ------------------------------
def save_results(result, base_path, img):
    """保存两种结果：目标检测和实例分割"""
    visualizer = ResultVisualizer(model)
    
    # 1. 保存目标检测结果（确保不包含掩膜）
    det_txt_path = f"{base_path}.txt"
    det_img_path = f"{base_path}_det.jpg"
    
    with open(det_txt_path, 'w') as f:
        for bbox, score, label in zip(
            result.pred_instances.bboxes,
            result.pred_instances.scores,
            result.pred_instances.labels
        ):
            label_idx = int(label)
            class_name = VOC_CLASSES[label_idx] if label_idx < len(VOC_CLASSES) else 'unknown'
            f.write(f"{class_name} {score:.4f} {int(bbox[0])} {int(bbox[1])} {int(bbox[2])} {int(bbox[3])}\n")
    
    # 关键修改：使用仅包含边界框的可视化方法
    visualizer.draw_detection(img, result, det_img_path)
    
    # 2. 保存实例分割结果（如果存在掩膜）
    if hasattr(result.pred_instances, 'masks') and result.pred_instances.masks is not None:
        seg_txt_path = f"{base_path}_seg.txt"
        seg_img_path = f"{base_path}_seg.jpg"
        
        with open(seg_txt_path, 'w') as f:
            for i, (bbox, score, label, mask) in enumerate(zip(
                result.pred_instances.bboxes,
                result.pred_instances.scores,
                result.pred_instances.labels,
                result.pred_instances.masks
            )):
                label_idx = int(label)
                class_name = VOC_CLASSES[label_idx] if label_idx < len(VOC_CLASSES) else 'unknown'
                f.write(f"Object {i}:\n")
                f.write(f"  Class: {class_name}\n")
                f.write(f"  Score: {score:.4f}\n")
                f.write(f"  BBox: {bbox.int().tolist()}\n\n")
        
        visualizer.draw_segmentation(img, result, seg_img_path)
        
# ------------------------------
# 主流程
# ------------------------------
def main():
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 开始批量处理图片")
    
    try:
        # 初始化模型
        global model
        model = init_model()
        
        # 处理每张图片
        for img_file in os.listdir(input_dir):
            if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            start_time = time.time()
            img_path = os.path.join(input_dir, img_file)
            base_name = os.path.splitext(img_file)[0]
            base_path = os.path.join(output_dir, base_name)
            
            print(f"\n[处理] {img_file}")
            
            # 读取和预处理图像
            img = mmcv.imread(img_path, channel_order='rgb')
            if max(img.shape[:2]) > max_image_size:
                scale = max_image_size / max(img.shape[:2])
                img = mmcv.imrescale(img, scale=scale)
            
            # 推理和处理结果
            result = inference_detector(model, img)
            processed_result = process_results(result, img.shape)
            
            # 保存结果
            save_results(processed_result, base_path, img)
            
            print(f"[完成] 耗时: {time.time()-start_time:.2f}s")
            
    except Exception as e:
        print(f"\n[错误] {str(e)}")
    finally:
        print(f"\n[结束] 所有处理完成")

if __name__ == '__main__':
    main()