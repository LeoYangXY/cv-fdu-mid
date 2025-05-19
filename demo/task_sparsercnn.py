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
config_path = 'configs/sparse_rcnn/sparse-rcnn_r50_fpn_1x_coco.py'
checkpoint_path = 'work_dirs/sparse-rcnn_r50_fpn_1x_voc/good_checkpoint.pth'
input_dir = 'data/new_pic'
output_dir = 'data/sparse_rcnn/result'
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
# 结果处理 (适配SparseRCNN特性)
# ------------------------------
def process_results(results, img_shape):
    """处理SparseRCNN的输出结果"""
    h, w = img_shape[:2]
    pred = results.pred_instances
    
    # SparseRCNN特有处理：确保proposal数量不超过设定值
    if len(pred) > num_proposals:
        keep = pred.scores.topk(num_proposals)[1]
        pred = pred[keep]
    
    # 坐标修正
    bboxes = pred.bboxes.clone()
    bboxes[:, 0::2] = torch.clamp(bboxes[:, 0::2], 0, w)
    bboxes[:, 1::2] = torch.clamp(bboxes[:, 1::2], 0, h)
    
    # 过滤低分检测
    keep_mask = pred.scores > score_threshold
    
    filtered = DetDataSample()
    filtered.pred_instances = InstanceData(
        bboxes=bboxes[keep_mask],
        scores=pred.scores[keep_mask],
        labels=pred.labels[keep_mask],
        masks=pred.masks[keep_mask] if hasattr(pred, 'masks') else None
    )
    return filtered

# ------------------------------
# 可视化工具 (适配SparseRCNN)
# ------------------------------
class SparseRCNNVisualizer:
    def __init__(self, model):
        self.visualizer = VISUALIZERS.build(model.cfg.visualizer)
        self.visualizer.dataset_meta = model.dataset_meta
    
    def draw_results(self, img, result, out_path):
        """绘制检测和分割结果"""
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
                'result',
                img,
                data_sample=result,
                draw_gt=False,
                show=False,
                out_file=out_path
            )

# ------------------------------
# 保存结果 (适配SparseRCNN)
# ------------------------------
def save_results(result, base_path, img):
    """保存检测和分割结果"""
    visualizer = SparseRCNNVisualizer(model)
    
    # 保存可视化结果
    vis_path = f"{base_path}_result.jpg"
    visualizer.draw_results(img, result, vis_path)
    
    # 保存检测结果文本
    txt_path = f"{base_path}.txt"
    with open(txt_path, 'w') as f:
        for i, (bbox, score, label) in enumerate(zip(
            result.pred_instances.bboxes,
            result.pred_instances.scores,
            result.pred_instances.labels
        )):
            class_name = COCO_CLASSES[int(label)] if int(label) < len(COCO_CLASSES) else 'unknown'
            f.write(f"Object {i}:\n")
            f.write(f"  Class: {class_name}\n")
            f.write(f"  Score: {score:.4f}\n")
            f.write(f"  BBox: {bbox.int().tolist()}\n\n")
    
    # 如果有掩膜，保存掩膜信息
    if hasattr(result.pred_instances, 'masks') and result.pred_instances.masks is not None:
        mask_path = f"{base_path}_mask.npy"
        np.save(mask_path, result.pred_instances.masks.cpu().numpy())

# ------------------------------
# 主流程
# ------------------------------
def main():
    print(f"[{time.strftime('%H:%M:%S')}] 🚀 开始批量处理图片")
    
    try:
        # 初始化模型
        global model, num_proposals
        model = init_model()
        num_proposals = model.test_cfg.rcnn.max_per_img
        
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