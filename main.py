import torch
import cv2
import os
from pathlib import Path
from ultralytics import YOLO

# Function to load the model
def load_model(weights_path):
    model = YOLO(weights_path)  # Load custom-trained model
    return model

# Function to make predictions on a single image
def predict_image(model, image_path, conf_threshold=0.01):
    # Load image
    img = cv2.imread(image_path)
    
    # Perform inference
    results = model(img)
    
    # Get the first result (the first image)
    result = results[0]
    
    # Access the detected boxes and their confidence scores
    boxes = result.boxes
    confidences = boxes.conf
    labels = boxes.cls

    # Filter predictions by confidence threshold
    filtered_boxes = boxes[confidences >= conf_threshold]
    filtered_confidences = confidences[confidences >= conf_threshold]
    filtered_labels = labels[confidences >= conf_threshold]

    # Display and save results
    for box, conf, label in zip(filtered_boxes, filtered_confidences, filtered_labels):
        # Get the coordinates of the bounding box (xyxy format)
        x1, y1, x2, y2 = box.xyxy[0].int()  # Convert to integer values
        
        # Get class name
        class_name = model.names[int(label)]
        
        # Draw bounding box and label
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
        label_text = f"{class_name} {conf:.2f}"
        cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the image with bounding boxes
    cv2.imshow('Predictions', img)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()

    # Save the result to disk
    output_path = os.path.join('predictions', Path(image_path).name)
    os.makedirs('predictions', exist_ok=True)
    cv2.imwrite(output_path, img)
    print(f"Predictions saved to {output_path}")

# Main function
def main():
    # Path to the trained model weights file
    weights_path = 'Fracture_V2.pt'  # Change this to your model file's path
    image_path = 's.jpg'  # Path to your single test image

    # Load model
    model = load_model(weights_path)

    # Predict on the single image with a low confidence threshold (0.01)
    predict_image(model, image_path, conf_threshold=0.01)

if __name__ == '__main__':
    main()
