 # depth_estimation.py

from transformers import DPTForDepthEstimation, DPTFeatureExtractor
import torch
import numpy as np

def load_depth_model():
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
    feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
    return model, feature_extractor

def estimate_depth(model, feature_extractor, frame):
    inputs = feature_extractor(images=frame, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        depth = outputs.predicted_depth.squeeze().cpu().numpy()
    return depth

