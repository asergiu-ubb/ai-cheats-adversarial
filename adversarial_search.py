import torch
import os
import random
from glob import glob
from PIL import Image
from torchvision import transforms
from ultralytics import YOLOv10
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import json
import time
import concurrent.futures
from functools import partial
from ultralytics import YOLO
from salience_detr.salience_detr_resnet50_800_1333 import model
from util.utils import load_state_dict
from files import IMAGES

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
CLASS_ID = 0
SALIENCE_CLASS_ID = 1


def load_models():
    yolo10 = YOLOv10('best_yolov10.pt')
    yolo8 = YOLO('best_yolov8.pt')
    salience_weights = torch.load("best_salience.pth")
    salience_model = model
    load_state_dict(salience_model, salience_weights)
    salience_model = salience_model.eval().cuda()
    
    return yolo10, yolo8, salience_model


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()


def load_labels(label_path):
    # Load labels for each image based on txt files corresponding to image files
    label_files = glob(os.path.join(label_path, '*.txt'))
    image_labels = {}
    
    for file in label_files:
        class_ids = set()
        image_id = os.path.splitext(os.path.basename(file))[0]
        with open(file) as f:
            labels = f.read().strip().split('\n')
            labels = [line.split() for line in labels]
            try:
                for label in labels:
                    class_ids.add(int(label[0]))
                image_labels[image_id] = class_ids
            except:
                pass
    
    return image_labels

def load_and_transform_image(img_path, image_labels, transform):
    img_id = os.path.splitext(os.path.basename(img_path))[0]
    image = transform(Image.open(img_path).convert('RGB'))
    label = image_labels.get(img_id, [])
    return image, label

def load_random_images(path, batch_size,image_labels, class_id):
    # Gather all image files
    all_images = glob(os.path.join(path, '*.[jp][pn]g'))

    target_images = [img for img in all_images if class_id in image_labels.get(os.path.splitext(os.path.basename(img))[0], [])]
    # Randomly select images
    selected_images = random.sample(target_images, min(batch_size, len(target_images)))
    print(selected_images)
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Size expected by the model
        transforms.ToTensor()
    ])
    
    
    
    images_tensor = []
    labels = []
    # Load and transform images
    with concurrent.futures.ThreadPoolExecutor() as executor:
        task = partial(load_and_transform_image, image_labels=image_labels, transform=transform)
        results = executor.map(task, selected_images)
        
    for image, label in results:
        images_tensor.append(image)
        labels.append(label)
    
    images_tensor = torch.stack(images_tensor)
    return images_tensor, labels

def get_perturbation(path):
    with open(path, 'r') as f:
        perturbation_array = json.load(f)
    return perturbation_array

def load_images(path, batch_size,image_labels, class_id, images):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Size expected by the model
        transforms.ToTensor()
    ])
    
    images_tensor = []
    labels = []
    # Load and transform images
    with concurrent.futures.ThreadPoolExecutor() as executor:
        task = partial(load_and_transform_image, image_labels=image_labels, transform=transform)
        results = executor.map(task, images)
        
    for image, label in results:
        images_tensor.append(image)
        labels.append(label)
    
    images_tensor = torch.stack(images_tensor)
    return images_tensor, labels

def get_yolo_results(model, images, class_id):
    outputs = model(images,verbose=False)
    #print([ set(prediction.boxes.cls.detach().cpu().numpy().astype(np.uint8)) for prediction in outputs])
    if any([class_id in set(prediction.boxes.cls.detach().cpu().numpy().astype(np.uint8)) for prediction in outputs]):
        return False
    return True

def get_yolo_results_all_correct(model, images, class_id):
    outputs = model(images,verbose=False)

    if all([class_id in set(prediction.boxes.cls.detach().cpu().numpy().astype(np.uint8)) for prediction in outputs]):
        return True
    return False

def get_salience_results(model,images,class_id):
    outputs = model(images.cuda())
    filtered_labels_set = set()
    for prediction in outputs:
        labels = prediction['labels']
        scores = prediction['scores']
        
        high_score_indices = scores >= 0.5
        
        filtered_labels = labels[high_score_indices]
        
        filtered_labels_set.update(filtered_labels.tolist())
    if SALIENCE_CLASS_ID in filtered_labels_set:
        return False
    return True 


def get_salience_results_all_correct(model,images,class_id):
    outputs = model(images.cuda())
    for prediction in outputs:
        labels = prediction['labels']
        scores = prediction['scores']
        
        high_score_indices = scores >= 0.5
        
        filtered_labels = labels[high_score_indices]
        if SALIENCE_CLASS_ID not in filtered_labels.tolist():
            return False
    return True 

def predict_images(yolov10, yolov8, salience, images_tensor, labels, class_id):
    with torch.no_grad():
        images_chunks = images_tensor.chunk(1)
        #suma = 0
        for image_chunk in images_chunks:
            yolov10_results = get_yolo_results(yolov10,image_chunk,class_id)
            if not yolov10_results:
                return False
            yolov8_results = get_yolo_results(yolov8,image_chunk,class_id)
            if not yolov8_results:
                return False
            salience_results = get_salience_results(salience, image_chunk, class_id)
            if not salience_results:
                return False
            
    return True

def are_all_images_correct(yolov10, yolov8, salience, images_tensor, labels, class_id):
    with torch.no_grad():
        salience_results = get_salience_results_all_correct(salience, images_tensor, class_id)
        if not salience_results:
            return False
        yolov10_results = get_yolo_results_all_correct(yolov10,images_tensor,class_id)

        if not yolov10_results:
            return False

        yolov8_results = get_yolo_results_all_correct(yolov8,images_tensor,class_id)
        if not yolov8_results:
            return False

    return True

def resize_with_padding(array, new_shape, constant_value=0):
    # Calculate the padding needed for each dimension
    pad_height = max(0, new_shape[0] - array.shape[0])  # Pad rows
    pad_width = max(0, new_shape[1] - array.shape[1])  # Pad columns

    # Define padding for top, bottom, left, and right
    padding = ((0, pad_height), (0, pad_width))

    # Apply padding to the array
    padded_array = np.pad(array, pad_width=padding, mode='constant', constant_values=constant_value)

    return padded_array

def create_perturbation_matrix(array, m):
    n = len(array)
    sqrt_n = int(np.sqrt(n))
    
    
    matrix = np.array(array).reshape((sqrt_n, sqrt_n))
    
    
    # Calculate the repetition factor needed to reach or exceed size m x m
    rep_factor = m // sqrt_n
    
    # Tile the image
    tiled_matrix = np.tile(matrix, (rep_factor, rep_factor))
    tiled_matrix = resize_with_padding(tiled_matrix, (640,640))
    tiled_matrix = 1 - tiled_matrix
    return tiled_matrix

def count_objects(outputs):
    # Count objects in each image
    counts = [output['instances'].num_instances() for output in outputs]
    return counts

def apply_perturbation(inputs, perturbations, color=(1, 0, 0)):
    batch_size, channels, height, width = inputs.shape
    mask = (perturbations == 0)
    
    for i in range(batch_size):
        for c in range(channels):
            inputs[i, c, :, :][mask] = color[c]
    
    return inputs

def main():
    PATH = "F://dizertatie//yolov10//yolov10-36efe34fe1126a4ce639403f1ce16fa0edbc16fd//datasets//apex//train//images//"
    PATH_LABELS = "F://dizertatie//yolov10//yolov10-36efe34fe1126a4ce639403f1ce16fa0edbc16fd//datasets//apex//train//labels//"
    BATCH_SIZE = 16
    max_pixels_changed = 7
    size = 16
    
    yolov10, yolov8, salience = load_models()
    image_labels = load_labels(PATH_LABELS)
    image_tensors_batch = torch.Tensor([])
    labels_batch = []
    images_tensor, labels = load_images(PATH, BATCH_SIZE, image_labels, CLASS_ID, IMAGES)
    for i in range(100000):
        perturbation_array = [1] * max_pixels_changed + [0] * (size-max_pixels_changed)
        random.shuffle(perturbation_array)
        perturbation = create_perturbation_matrix(perturbation_array, 640)
        inputs = images_tensor.clone()
        inputs = apply_perturbation(inputs, perturbation, color=(0, 1, 0))
        predictions = predict_images(yolov10, yolov8, salience, inputs, labels, CLASS_ID)
    
        if predictions:
            print(f"Perturbation found at:{i}")
            with open(f'perturbation_results_adv/data_{i}.json', 'w') as f:
                json.dump(perturbation_array, f)
        if i % 10 == 0:
            print(f"At step:{i}")

if __name__ == '__main__':
    main()
    input()