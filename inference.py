from ultralytics import YOLO
import cv2



# CSPDarknet wala
model_cspdarknet53 = YOLO("runs/detect/fire_smoke_model4/weights/best.pt")

# print(model)
# image_path = "image.png"
image_path = "dataset/images/train/Datacluster Fire and Smoke Sample (5).jpg"
# image_path = "server/saved_frames/trrJwmF4hXEGcGcjAAAD/trrJwmF4hXEGcGcjAAAD_20251110_223806_724847.jpg"

# Run inference
results = model_cspdarknet53(image_path, conf=0.8, save=True, show=False)




# custom wala
model = YOLO("runs/detect/fire_smoke_model5/weights/best.pt")

# print(model)
image_path = "dataset/images/train/Datacluster Fire and Smoke Sample (5).jpg"
# image_path = "image.png"
# image_path = "server/saved_frames/trrJwmF4hXEGcGcjAAAD/trrJwmF4hXEGcGcjAAAD_20251110_223806_724847.jpg"

# Run inference
results = model(image_path, conf=0.8, save=True, show=False)
