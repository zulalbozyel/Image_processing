# Object Detection and Language Learning Application

This project provides the foundation for a real-time object detection and language learning application. It captures live images from a camera, detects objects, translates their names into Turkish, and stores the results both as text and visually styled image cards.

## Technologies Used

### OpenCV (cv2)
Used for capturing real-time images from the camera, drawing bounding boxes around detected objects, displaying windows, and performing basic image processing operations.

### YOLO (Ultralytics)
Used for real-time object detection. The project includes multiple model weights:
- YOLOv8 (yolov8n.pt, yolov8s.pt)
- YOLO11 (yolo11m.pt)

### deep_translator
Used to translate object names detected by YOLO (in English) into Turkish using the GoogleTranslator module.

### Pillow (PIL - Python Imaging Library)
Used in the photo_card.py file to generate polaroid-style image cards. This includes adding text, formatting layouts, and applying visual effects such as spacing and shadows.

### NumPy
Used for handling array operations and converting between OpenCV (BGR) and Pillow (RGB) image formats.

## Project Overview

This project performs the following operations:
- Captures real-time images from a camera
- Detects objects using deep learning models
- Translates detected object names from English to Turkish
- Saves translated words into a text file as a vocabulary list
- Generates visually styled image cards for each detected object

In summary, this project represents a combination of object recognition and language learning functionalities.

## Notes

Model weight files (.pt) are not included in the repository due to size limitations. These files should be downloaded separately and placed in the appropriate directory before running the project.

## Contributing

Contributions are welcome. You can fork the repository and submit pull requests for improvements or additional features.

## Contact

For any questions or feedback, please feel free to get in touch.
