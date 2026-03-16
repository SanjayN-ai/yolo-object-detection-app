import numpy as np
from PIL import Image
from ultralytics import YOLO

class YolosHelper:
    def __init__(self, model_path="yolov8n-seg.pt"):
        # Load the YOLOv8 segmentation model
        self.model = YOLO(model_path)

    def predict_and_annotate(self, image: Image.Image):
        """
        Runs YOLOv8 segmentation on the input PIL Image.
        Returns the annotated image (as a PIL Image) and a dictionary
        of detected object names with their counts.
        """
        # Convert PIL image to OpenCV format (BGR) for potentially better compatibility,
        # though Ultralytics YOLO handles PIL images directly too.
        # We will pass the PIL image directly.
        
        # Run inference
        results = self.model(image)
        
        # The result object contains the original image, boxes, masks, and names.
        result = results[0]
        
        # Get annotated image (BGR numpy array)
        annotated_bgr = result.plot()
        
        # Convert BGR to RGB using numpy slicing (no cv2 needed)
        annotated_rgb = annotated_bgr[:, :, ::-1]
        annotated_image = Image.fromarray(annotated_rgb)
        
        # Extract detected object names and counts
        detected_counts = {}
        if result.boxes is not None:
            # result.boxes.cls contains the class indices of the detections
            class_indices = result.boxes.cls.cpu().numpy().astype(int)
            names_dict = result.names
            
            for cls_idx in class_indices:
                class_name = names_dict[cls_idx]
                detected_counts[class_name] = detected_counts.get(class_name, 0) + 1
                
        return annotated_image, detected_counts
