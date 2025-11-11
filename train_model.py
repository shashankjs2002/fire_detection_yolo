from ultralytics import YOLO

# Load a pretrained YOLOv8 model
# model = YOLO('yolov8n.pt')  # n=nano (fastest), s=small, m=medium, l=large
# model = YOLO('yolov8n_mobilenet.yaml')
model = YOLO('fire_lite.yaml')


print(model)
# Train the model
results = model.train(
    data='data.yaml',           # Path to your data.yaml
    epochs=100,                  # Number of training epochs (increase for better results)
    imgsz=640,                  # Image size
    batch=1,                   # Batch size (reduce if you get memory errors)
    name='fire_smoke_model',    # Name for this training run
    patience=20,                # Early stopping patience
    device='cpu'                # Use 'cpu' or '0' for GPU
)

print("Training completed!")
print(f"Best model saved at: {model.trainer.best}")


# ----------------firedetect_lite-----------------

# from ultralytics import YOLO

# # Load custom model
# # model = YOLO('firedetect_lite.yaml')
# model = YOLO('fire_lite.yaml')

# # Train
# results = model.train(
#     data='data.yaml',
#          epochs=100,
#             imgsz=640,
#             batch=16,
            
#             # Fire-specific augmentation
#             hsv_h=0.015,
#             hsv_s=0.7,
#             hsv_v=0.4,
#             degrees=10,
#             translate=0.1,
#             scale=0.5,
#             mosaic=1.0,
#             mixup=0.1,
            
#             # Optimizer
#             optimizer='AdamW',
#             lr0=0.015,      # Higher LR for lighter model
#             lrf=0.01,
#             momentum=0.937,
#             weight_decay=0.0005,
            
#             # Loss weights
#             box=7.5,
#             cls=0.5,
#             dfl=1.5,
            
#             # Settings
#             name='fire_lite',
#             patience=50,
#             device='cpu',   # Change to '0' for GPU
#             workers=4,
#             cache=False,
#             plots=True,
#             save=True,
#             verbose=True,

#         )
        