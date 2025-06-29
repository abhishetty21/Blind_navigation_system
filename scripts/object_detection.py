import torch
import torchvision
from torchvision.transforms import functional as F

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

def load_model():
    model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
    model.eval()
    return model

def detect_objects(model, frame):
    # Convert the frame to tensor
    frame_tensor = F.to_tensor(frame).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(frame_tensor)[0]
    
    # Map detected classes to COCO labels
    detected_objects = []
    for i in range(len(predictions['labels'])):
        label_index = predictions['labels'][i].item()  # Convert tensor to int
        label_name = COCO_INSTANCE_CATEGORY_NAMES[label_index] if label_index < len(COCO_INSTANCE_CATEGORY_NAMES) else "Unknown"
        score = predictions['scores'][i].item()
        box = predictions['boxes'][i].tolist()
        
        if score > 0.5:  # Confidence threshold (optional)
            detected_objects.append({
                'label': label_name,
                'score': score,
                'box': box
            })

    return detected_objects
