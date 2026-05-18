from ultralytics import YOLO
import time
import logging

logging.getLogger("ultralytics").setLevel(logging.ERROR)
#load a model
model = YOLO('yolov8x.pt')

start_time = time.time()
# model train

model.train(data="holes_v3.yaml",  #path to dataset YAML
    epochs=3,  # number of training epochs
    imgsz=640,  # training image size
    batch=8,
    device="0, 1, 2, 3, 4, 5, 6, 7",
    lr0=1e-5,
    optimizer="AdamW",
    project="~/root",
    name="holes_new_yolo",
    verbose=False,
    plots=False,
)

end_time = time.time()

total_seconds = end_time - start_time
hours = int(total_seconds // 3600)
minutes = int((total_seconds % 3600) // 60)
seconds = int(total_seconds % 60)

print(f"Training finished.")
print(f"Total training time: {hours}h {minutes}m {seconds}s "
      f"({total_seconds:.2f} seconds)")
'''
model.train(data="electrical_board.yaml",  #path to dataset YAML
    epochs=50,  # number of training epochs
    imgsz=640,  # training image size
    batch=16,
    device="0")
'''
