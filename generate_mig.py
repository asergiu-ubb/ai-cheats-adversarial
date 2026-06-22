import argparse
import logging
import os
import time

import numpy as np
import torch
from ultralytics import YOLO

from MIG import BCE_loss_boxes, get_preprocessor, get_universal_perturbation_mean_gradients
from eval_invisibility_cloak import tog_vanishing_with_update, our_attack_success
from eval_yolo import get_yolo_boxes, get_predicted_boxes, calculate_map
from test_invisibility_cloak_apex import FindNoise, tog_vanishing
from ultralytics import RTDETR

IMAGE_SIZE = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def join_basenames(paths):
    # Extract basenames from paths and join them with "_"
    basenames = [os.path.basename(os.path.dirname(path)) for path in paths]
    return "_".join(basenames)


def main(args):
    # Set up logging
    log_file = args.log_file
    log_file= log_file.replace("/log.txt", f'{args.experiment}/MIG_pretrained/log.txt')
    log_dir = os.path.dirname(log_file)
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    # Start of the experiment
    start_time = time.time()
    logger.info("Experiment started.")
    logger.info(f"Arguments: {args}")

    # Set up models
    logger.info("Loading models...")
    models_paths = [
        "models/yolo8n_640_all.pt",
        "models/yolo8s_640_all.pt",
        "models/yolo11n_640_all.pt",
        "models/yolo11s_640_all.pt",
        "models/yolo5s_640_all.pt",
        "models/yolo5n_640_all.pt"
    ]
    logger.info(f"DEVICE: {DEVICE}.")
    all_models = []
    for m_idx, model_path in enumerate(models_paths):
        logger.info(f"Loading {model_path}...")
        if "yolo" in model_path:
            model = YOLO(model_path, verbose=False).to(DEVICE)
        else:
            model = RTDETR(model_path).to(DEVICE)
        all_models.append(model)


    # Set up preprocessor
    batch_size = args.batch_size
    preprocessor = get_preprocessor(all_models[0], batch_size)
    baseline = torch.zeros((batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
    loss_fn = BCE_loss_boxes

    # Load datasets
    logger.info("Loading datasets...")
    running_percent = 1.0 # Adjust as needed
    train_imgs, train_labels = [], []
    for dataset_path in args.train_datasets:
        imgs, labels, _ = get_yolo_boxes(dataset_path, running_percent, img_size=IMAGE_SIZE)
        train_imgs.extend(imgs)
        train_labels.extend(labels)
    test_imgs, test_labels = [], []
    for dataset_path in args.test_datasets:
        imgs, labels, _ = get_yolo_boxes(dataset_path, running_percent, img_size=IMAGE_SIZE)
        test_imgs.extend(imgs)
        test_labels.extend(labels)

    # Generate or load initial perturbation
    epsilon = args.epsilon / 255.0

    logger.info("Generating initial perturbation using MIG...")
    iterations = 25
    momentum = 1
    order_of_approximation = 20
    initial_perturbation = get_universal_perturbation_mean_gradients(
        all_models, train_imgs, train_labels, epsilon, iterations, momentum, loss_fn, baseline, preprocessor, order_of_approximation, batch_size)
    init_perturbation_path = f"{log_file.replace('.log','')}_init_perturbation.pt"
    torch.save(initial_perturbation, init_perturbation_path)
    logger.info(f"Trained perturbation saved to {init_perturbation_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with specified parameters.")
    parser.add_argument("--train-datasets", nargs='+', required=True, help="List of paths for datasets to be used for training.")
    parser.add_argument("--test-datasets", nargs='+', required=True, help="List of paths for datasets to be used for testing.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for processing images.")
    parser.add_argument("--epsilon", type=float, default=10, help="Epsilon value for the perturbation (in pixel intensity units).")
    parser.add_argument("--log-file", default=f"runs/"+"/log.txt", help="Path to the log file.")
    parser.add_argument("--experiment")
    args = parser.parse_args()

    main(args)