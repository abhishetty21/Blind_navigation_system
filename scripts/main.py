import numpy as np
import cv2
import torch
from object_detection import load_model, detect_objects
from depth_estimation import load_depth_model, estimate_depth
from voice_output import initialize_voice_engine, speak_message

# Initialize models and engine
detection_model = load_model()
depth_model, depth_extractor = load_depth_model()
voice_engine = initialize_voice_engine()

def main():
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Object Detection
        detections = detect_objects(detection_model, frame)
        
        # Depth Estimation
        depth_map = estimate_depth(depth_model, depth_extractor, frame)
        
        # Process each detection
        for i, detection in enumerate(detections):
            box = detection['box']
            label = detection['label']
            score = detection['score']
            
            if score > 0.5:  # Adjust threshold as needed
                # Extract box coordinates
                x1, y1, x2, y2 = map(int, box)
                
                # Estimate distance (average depth within bounding box)
                obj_depth = np.mean(depth_map[y1:y2, x1:x2])
                
                # Generate voice message
                message = f"{label} is approximately {int(obj_depth)} cm away."
                speak_message(voice_engine, message)
                
                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label}: {int(obj_depth)} cm", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Show the frame
        cv2.imshow("Blind Navigation", frame)
        
        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
