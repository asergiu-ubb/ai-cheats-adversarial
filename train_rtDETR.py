import sys
import wandb
from ultralytics import RTDETR

ds = "csgo_apex_aim" #"csgo"
model_name = "rtdetr-x"

wandb.login(key="c1a00e18c0eb7f9707f86e25ab70caf609c05281", relogin=True)
wandb.init(project="blackBoxAdv", job_type="training",name=model_name+"_"+ds)

is_windows = sys.platform.startswith('win')
if is_windows:
    experiment = 'experiments/train_apex_aim.yaml'
else:
    if ds == "csgo":
        experiment = 'experiments/train_csgo.yaml'
    elif ds == "csgo_apex_aim":
        experiment = 'experiments/train_csgo_apex_aim.yaml'
    else:
        experiment = 'experiments/train_apex_aim_ubuntu.yaml'

model = RTDETR(f"{model_name}.pt")
model.train(data=experiment, epochs=50, batch=12, imgsz=640,workers=0, name=model_name+"_"+ds)