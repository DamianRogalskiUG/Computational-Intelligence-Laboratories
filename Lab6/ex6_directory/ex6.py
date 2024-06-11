import cv2
import numpy as np
import os
from keras.preprocessing.image import load_img, img_to_array

target_size = (255, 255)  # Target size for resizing images


def load_blobs():
    # Load images from the 'images' directory and convert them to blobs
    files = os.listdir("./images")
    blobs = []
    for f in files:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            img_path = os.path.join("./images", f)
            img = load_img(img_path, target_size=target_size)
            img_array = img_to_array(img)
            # Convert image array to blob
            blob = cv2.dnn.blobFromImage(
                img_array, 1 / 255.0, target_size, swapRB=True, crop=False
            )
            blobs.append((img_path, blob))
    return blobs


def load_class_names():
    # Load class names from file
    with open("coco.names", "r") as f:
        class_names = f.read().strip().split("\n")
    return class_names


if __name__ == "__main__":
    blobs = load_blobs()
    class_names = load_class_names()

    # Load YOLO model
    net = cv2.dnn.readNetFromDarknet("yolov3.cfg", "yolov3.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

    for img_path, blob in blobs:
        original_image = cv2.imread(img_path)
        (H, W) = original_image.shape[:2]

        # Set input blob for the network and forward pass
        net.setInput(blob)
        outputs = net.forward(ln)

        boxes = []
        confidences = []
        classIDs = []

        # Process YOLO output_images
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # Adjust this threshold as needed
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    startX = int(centerX - (width / 2))
                    startY = int(centerY - (height / 2))
                    endX = startX + width
                    endY = startY + height
                    boxes.append([startX, startY, endX, endY])
                    confidences.append(float(confidence))
                    classIDs.append(class_id)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        if len(indices) > 0:
            for i in indices.flatten():
                (startX, startY, endX, endY) = boxes[i]
                color = [int(c) for c in np.random.uniform(0, 255, size=(3,))]
                class_name = class_names[classIDs[i]]
                # Draw bounding box and put text
                cv2.rectangle(original_image, (startX, startY), (endX, endY), color, 2)
                text = f"{class_name} - {confidences[i]:.2f}"
                cv2.putText(
                    original_image,
                    text,
                    (startX, startY - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )
                print(f"{img_path} : {text}")

        output_path = os.path.join("./output_images", os.path.basename(img_path))
        cv2.imwrite(output_path, original_image)
        print(f"Output image saved to {output_path}")
