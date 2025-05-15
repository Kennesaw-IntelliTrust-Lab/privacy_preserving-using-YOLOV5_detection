import cv2
import numpy as np
import onnxruntime
import os

# Load ONNX model
session = onnxruntime.InferenceSession(
    "runs/train/exp13/weights/best_quantized.onnx",
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name

# Set threshold
conf_thres = 0.3

# Load test image(s)
source_folder = "/mnt/bst/hxu10/hxu10/chanti/dataset/images/test"
output_folder = "runs/detect/best_onnx_detect"
os.makedirs(output_folder, exist_ok=True)

# Loop over test images
for filename in os.listdir(source_folder):
    if not filename.endswith((".jpg", ".png", ".jpeg")):
        continue

    image_path = os.path.join(source_folder, filename)
    img_raw = cv2.imread(image_path)
    img = cv2.resize(img_raw, (640, 640))
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB + CHW
    img = np.ascontiguousarray(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Run inference
    outputs = session.run(None, {input_name: img})[0]

    # Postprocess (very basic): draw boxes for outputs > conf_thres
    for det in outputs[0]:  # shape (25200, 8)
        x1, y1, x2, y2, conf, cls = det[0], det[1], det[2], det[3], det[4], int(det[5])
        if conf > conf_thres:
            cv2.rectangle(img_raw, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img_raw, f"Class {cls} {conf:.2f}", (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save output
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, img_raw)
    print(f"✅ Saved: {output_path}")
