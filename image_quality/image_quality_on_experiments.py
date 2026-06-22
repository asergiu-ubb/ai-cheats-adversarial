import glob
import os
import sys
import cv2
import numpy
import numpy as np
import torch
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from tqdm import tqdm
from ultralytics.models import YOLO

sys.path.append("..")
sys.path.append("../..")

from MIG import get_preprocessor
from image_quality import compute_uqi, compute_snr
from utils.standardize import standardize_path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    epsilon_list = [4, 8, 16, 32]

    # root_dir_images = standardize_path("Z:/image_databases/adversarialAttack/all_640/")
    root_dir_images = standardize_path("Z:/image_databases/adversarialAttack/all/valid_small")

    quality_log_file = "new_quality_new_th_full_image_640_eps.txt"
    root_dir = standardize_path("Y:/blackboxadversarial/runs/")

    directories = [
        # "abelation_new_eps/abelation_new_eps/[init]_random_[pert]_get_adv_[only_rtdets]_False_[train_dataset]_all_640_[test_dataset]_all_640_[train]_yolo8n_640_all",
        "abelation_only_mig/[init]_MIG_[pert]_get_adv_[only_rtdets]_False_[train_dataset]_all_640_[test_dataset]_all_640_[train]_yolo8n_640_all",
    ]

    images_list = []
    images_paths = glob.glob(os.path.join(root_dir_images, "**/*.jpg"), recursive=True)[:3]

    for image_path in images_paths:
        images_list.append(cv2.imread(image_path))

    with open(quality_log_file, "a") as f:
        f.write("images " + str(len(images_list)) + "images\n")
        print("images", len(images_list), "images")

        for directory in directories:
            log_file = os.path.join(root_dir, directory, "log.txt")
            perturbation_path = os.path.join(root_dir, directory, "log.txt_trained_perturbation.pt")
            if not os.path.exists(perturbation_path):
                print(perturbation_path,"does not exist")
                continue

            f.write("$$$ "+ directory+"\n")
            print("$$$ ", directory)

            for epsilon in epsilon_list:
                perturbation_value = torch.load(perturbation_path).to("cpu")
                if "MIG" in directory:
                    threshold = 0.01317
                    perturbation_value = torch.where(perturbation_value.abs() < threshold, torch.tensor(0.0), perturbation_value)
                    perturbation_value = torch.sign(perturbation_value) * (epsilon / 255)
                    # np.histogram(perturbation_value.detach().cpu().numpy())

                perturbation_value = torch.clamp(perturbation_value,-epsilon / 255, epsilon / 255)
                perturbation_value_np = perturbation_value.detach().to("cpu").numpy()
                metrics = []

                for img_idx in tqdm(range(0, len(images_list))):

                    batch_imgs = images_list[img_idx]
                    current_img = batch_imgs / 255.
                    current_img = np.transpose(current_img,(2, 0, 1))

                    start_y = (current_img.shape[1] - perturbation_value.shape[1]) // 2
                    start_x = (current_img.shape[2] - perturbation_value.shape[2]) // 2
                    current_img_adv = np.array(current_img)

                    current_img_adv[:, start_y:start_y + perturbation_value.shape[2], start_x:start_x + perturbation_value.shape[1] ] +=  perturbation_value_np
                    current_img_adv = numpy.clip(current_img_adv, 0.0, 1.0)

                    current_img = (current_img*255).astype(np.uint8)
                    current_img_adv = (current_img_adv*255).astype(np.uint8)


                    ssim_value = structural_similarity(current_img, current_img_adv,channel_axis=0, multichannel=True, data_range=current_img.max() - current_img.min())
                    psnr_value = peak_signal_noise_ratio(current_img, current_img_adv)
                    uqi_value = compute_uqi(current_img, current_img_adv)
                    snr_value = compute_snr(current_img, current_img_adv)
                    # print("ssim_value",ssim_value,"uqi_value",uqi_value)
                    metrics.append({
                        'ssim': ssim_value,
                        'psnr': psnr_value,
                        'uqi': uqi_value,
                        'snr': snr_value,
                    })

                    # display write
                    out_display_dir = "D:/school/blackBoxArticle/quality_640_pes"
                    perturbed_file_path = os.path.join(out_display_dir, f"pert_inv_{img_idx}_eps{epsilon}.jpg")
                    perturbed_file_txt_path = perturbed_file_path.replace(".jpg", f"_eps{epsilon}.txt")
                    orig_file_path = os.path.join(out_display_dir, f"clean{img_idx}_eps{epsilon}.jpg")
                    os.makedirs(out_display_dir, exist_ok=True)
                    if "MIG" in directory:
                        perturbed_file_path = os.path.join(out_display_dir, f"pert_our_{img_idx}_eps{epsilon}.jpg")
                        cv2.imwrite(orig_file_path, batch_imgs[start_y:start_y + perturbation_value.shape[2], start_x:start_x + perturbation_value.shape[1],:])
                        # cv2.imwrite(orig_file_path, batch_imgs)
                    current_img_adv_display = np.transpose(current_img_adv, (1, 2, 0))

                    with open(perturbed_file_txt_path, "w") as fq:
                        fq.write(f"SSIM: {'%.4f' % ssim_value} UQI: {'%.4f' % uqi_value}")
                    cv2.imwrite(perturbed_file_path, current_img_adv_display[start_y:start_y + perturbation_value.shape[2], start_x:start_x + perturbation_value.shape[1],:])
                    # cv2.imwrite(perturbed_file_path, current_img_adv_display)

                    # display imshow
                    # cv2.namedWindow("batch_imgs", cv2.WINDOW_NORMAL)
                    # cv2.namedWindow("current_img_adv_display", cv2.WINDOW_NORMAL)
                    # cv2.imshow("batch_imgs", batch_imgs)
                    # cv2.imshow("current_img_adv_display", current_img_adv_display)
                    # cv2.waitKey(0)

                ssim_values = [m['ssim'] for m in metrics]
                psnr_values = [m['psnr'] for m in metrics]
                uqi_values = [m['uqi'] for m in metrics]
                snr_values = [m['snr'] for m in metrics]

                f.write(" EPS "+str(epsilon)+"\n")
                print(" EPS ",epsilon)

                f.write(f"SSIM  Average: {np.mean(ssim_values):.4f}"+"\n")
                f.write(f"PSNR  Average: {np.mean(psnr_values):.2f} dB"+"\n")
                f.write(f"UQI  Average: {np.mean(uqi_values):.4f}"+"\n")
                f.write(f"SNR  Average: {np.mean(snr_values):.2f} dB"+"\n")
                # f.write(f"MSE Average: {np.mean(mse_values):.2f}"+"\n")
                f.write("\n")
                print(f"SSIM  Average: {np.mean(ssim_values):.4f}")
                print(f"PSNR  Average: {np.mean(psnr_values):.2f} dB")
                print(f"UQI  Average: {np.mean(uqi_values):.4f}")
                print(f"SNR  Average: {np.mean(snr_values):.2f} dB")
                # print(f"MSE Average: {np.mean(mse_values):.2f}")