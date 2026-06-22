import glob
import os

import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio, mean_squared_error
from ultralytics.models import YOLO

from MIG import get_preprocessor
from image_quality import compute_uqi, compute_snr

if __name__ == '__main__':

    # root_dir = "D:/work/blackboxadversarial/runs/run_21_oct"
    root_dir = "D:/work/blackboxadversarial/runs/run_23_oct_transf"
    # directories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    directories = [
    "[init]_random_[pert]_get_adv_[train_dataset]_apex_aim_[test_dataset]_apex_aim_[train]_yolov8n",
    # "[init]_MIG_[pert]_get_adv_sign_[train_dataset]_apex_aim_[test_dataset]_apex_aim_[train]_yolov8n"
    ]

    root_dir_images = "D:/work/image_databases/adversarialAttack"
    images_root_path = [
        os.path.join(root_dir_images,"aim","test"),
                   # os.path.join(root_dir_images,"CSGO","test"),
                   # os.path.join(root_dir_images,"apex","test")
                        ]
    images = []
    all_images_paths =[]
    for image_root_path in images_root_path:
        images_paths = glob.glob(os.path.join(image_root_path,"**/*.jpg"), recursive=True)
        all_images_paths.extend(images_paths)
        for image_path in images_paths:
            images.append(cv2.imread(image_path))

    batch_size = 1
    model_path = "../models/yolov8n_csgo_apex_aim.pt"
    model = YOLO(model_path, verbose=False).to("cpu")
    preprocessor = get_preprocessor(model, batch_size)

    max_ssim = 0.0
    best_ssim_pert_img = ""

    for experiment_directory in directories:
        epsilon_list = [0]
        if "_sign_" in experiment_directory:
            epsilon_list = [16]
        # perturbation_path = os.path.join(root_dir, experiment_directory, "log.txt_trained_perturbation.pt")
        perturbation_path = "../runs/Best-Succ-0.9143-BS-16-LR-0.001.pt"
        if not os.path.exists(perturbation_path):
            continue
        print("$$$ ", experiment_directory)
        perturbation_value = torch.load(perturbation_path).to("cpu")
        perturbation_value = torch.tile(perturbation_value[0], (2, 2))
        print("perturbation loaded")
        perturbation_value.detach().cpu().numpy()
        for epsilon in epsilon_list:
            if "_sign_" in experiment_directory:
                perturbation_value = torch.sign(perturbation_value) * (epsilon / 255)

            # pert_sign = torch.sign(perturbation_value)
            # pert_sign = torch.clamp(pert_sign, 0, 1.0)
            # pert_sign = pert_sign * 255
            # pert_sign = pert_sign.detach().to("cpu").numpy().astype(np.uint8)
            # c1 = pert_sign[0]
            # c2 = pert_sign[1]
            # c3 = pert_sign[2]
            # cv2.imwrite( os.path.join(root_dir, experiment_directory, "pert_1") + ".png", c1)
            # cv2.imwrite( os.path.join(root_dir, experiment_directory, "pert_2") + ".png", c2)
            # cv2.imwrite( os.path.join(root_dir, experiment_directory, "pert_3") + ".png", c3)
            eps_dir = "sign_"+str(epsilon)
            os.makedirs(os.path.join(root_dir, experiment_directory, eps_dir), exist_ok=True)

            for start_batch_idx in range(0, len(images), batch_size):
                print(start_batch_idx,"/",len(images))
                batch_imgs = images[start_batch_idx:start_batch_idx + batch_size]
                batch_imgs_preprocessed = preprocessor(batch_imgs)
                batch_imgs_preprocessed_adv = batch_imgs_preprocessed + perturbation_value
                batch_imgs_preprocessed_adv = torch.clamp(batch_imgs_preprocessed_adv, 0, 1.0)
                batch_imgs_preprocessed_adv_np = batch_imgs_preprocessed_adv.detach().to("cpu").numpy()

                perturbed_images_rescaled = batch_imgs_preprocessed_adv_np
                perturbed_images_rescaled = (perturbed_images_rescaled * 255).astype(np.uint8)

                for img_idx_in_batch in range(len(batch_imgs)):
                    img_idx_final = start_batch_idx + img_idx_in_batch
                    # print(all_images_paths[img_idx_final])
                    base_name = os.path.basename(all_images_paths[img_idx_final])
                    out_file_path = os.path.join(root_dir, experiment_directory, eps_dir, base_name)+".jpg"
                    perturbed_image_rescaled = np.transpose(perturbed_images_rescaled[img_idx_in_batch], (1, 2, 0))

                    # current_img = np.transpose(torch.clamp(batch_imgs_preprocessed[img_idx_in_batch],0,1).detach().to("cpu").numpy(), (1, 2, 0))
                    current_img = np.transpose(batch_imgs_preprocessed[img_idx_in_batch].detach().to("cpu").numpy(), (1, 2, 0))
                    current_img_adv = np.transpose(batch_imgs_preprocessed_adv_np[img_idx_in_batch], (1, 2, 0))

                    ssim_value = structural_similarity(current_img, current_img_adv, multichannel=True, channel_axis=2,
                                                       data_range=current_img.max() - current_img.min())

                    psnr_value = peak_signal_noise_ratio(current_img, current_img_adv)
                    uqi_value = compute_uqi(current_img, current_img_adv)
                    snr_value = compute_snr(current_img, current_img_adv)
                    mse_value = mean_squared_error(current_img, current_img_adv)

                    # if ssim_value > max_ssim:
                    #     max_ssim = ssim_value
                    #     best_ssim_pert_img = all_images_paths[img_idx_final]
                    print("sim_value %.4f" % ssim_value, "SPNR: %.2f" % psnr_value, "UQI: %.2f" % uqi_value,
                              "SNR:%.2f" % snr_value, best_ssim_pert_img)

                    perturbed_image_rescaled = cv2.cvtColor(perturbed_image_rescaled, cv2.COLOR_RGB2BGR)
                    concatenated_image = np.concatenate((perturbed_image_rescaled, batch_imgs[img_idx_in_batch]), axis=1)

                    # cv2.imwrite(out_file_path, concatenated_image)
                    # print("write", out_file_path)
                    # cv2.imshow("concatenated_image", concatenated_image)
                    # cv2.imshow("img", batch_imgs[img_idx_in_batch])
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()