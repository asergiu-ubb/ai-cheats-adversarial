import sys
import wandb
from ultralytics import YOLO
# Initialize a Weights & Biases run

model_name = "yolo11n"

# wandb.login(key="c1a00e18c0eb7f9707f86e25ab70caf609c05281", relogin=True)
# wandb.init(project="blackBoxAdv", job_type="training",name=model_name,tags=model_name,notes=model_name,id=model_name)

is_windows = sys.platform.startswith('win')
if is_windows:
    experiment = 'experiments/train_apex_aim.yaml'
else:
    experiment = 'experiments/train_apex_aim_ubuntu.yaml'

model = YOLO(f"{model_name}.pt")
model.train(data=experiment, epochs=50, batch=64, imgsz=640,workers=0)