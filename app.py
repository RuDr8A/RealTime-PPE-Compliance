import cv2
import numpy as np
from tensorflow.keras.models import load_model # type: ignore


# 1. Loading the trained Deep Learning model
print("Loading mask detection model...")
model = load_model('ppe_mask_model.keras')

class_names = ['Incorrect Mask', 'With Mask', 'No Mask']
colors = [(0, 255, 255), (0, 255, 0), (0, 0, 255)]

# 2. Initializing YuNet Face Detector
print("Loading YuNet face detector...")
face_detector = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2026may.onnx", 
    "",
    (320, 320), 
    score_threshold=0.7, 
    nms_threshold=0.3,
    top_k=5000
)

# 3. Initializing the webcam feed
print("Starting webcam... Press 'q' on your keyboard to quit.")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    frame = cv2.flip(frame, 1)
        
    # YuNet requires the input size to match current frame dimensions
    height, width, _ = frame.shape
    face_detector.setInputSize((width, height))
    
    # Detecting all faces currently in the frame
    _, faces = face_detector.detect(frame)
    
    # Processing each detected face one by one
    if faces is not None:
        for face in faces:
            # YuNet returns a 15-element array per face. The first 4 are the bounding box.
            box = face[0:4].astype(np.int32)
            x, y, w, h = box[0], box[1], box[2], box[3]
            
        
            # Expanding the bounding box slightly to capture full mask context
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.2)
            
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(width, x + w + pad_x)
            y2 = min(height, y + h + pad_y)
            
            # Extracting just the face's pixel matrix using guarded coordinates
            face_roi = frame[y1:y2, x1:x2]
            
            # Skiping if the ROI is empty or invalid
            if face_roi.size == 0 or face_roi.shape[0] == 0 or face_roi.shape[1] == 0:
                continue
            
            ## Converting BGR to RGB and resize
            face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224, 224))
            
            # Adding batch dimension and ensuring it is a float array
            face_array = np.expand_dims(face_resized, axis=0).astype(np.float32)
            
            # Feeding the RAW face directly into the model (the model preprocesses it internally!)
            predictions = model.predict(face_array, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = predictions[0][class_idx] * 100
            
            # Setting up the dynamic labels and colors
            label = f"{class_names[class_idx]} ({confidence:.1f}%)"
            color = colors[class_idx]
            
            # Drawing the bounding box and text overlay onto the live frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, max(y - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
    # Displaying the final processed frame
    cv2.imshow('Real-Time PPE Compliance Monitor', frame)
    
    # Listening for the 'q' key to cleanly exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleaning up the camera and close windows when done
cap.release()
cv2.destroyAllWindows()