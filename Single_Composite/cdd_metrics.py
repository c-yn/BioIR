########################################
## Evaluation code follows OneRestore.
########################################
import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import mean_squared_error as compare_mse
from skimage.metrics import structural_similarity as compare_ssim
# Modified function to add progress display using tqdm for better progress tracking
from tqdm import tqdm
import pandas as pd
from basicsr.utils.options import parse
import argparse

# Updated function with progress display for PSNR and SSIM calculation
def calculate_psnr_ssim_with_progress(clear_folder, degradation_types, methods, degradation_path, win_size=7):
    # Get list of all clear images
    img_list = [img for img in os.listdir(clear_folder) if img.endswith('.png')]
    
    # Initialize matrices to store mean PSNR and SSIM values
    psnr_matrix = np.zeros((len(methods), len(degradation_types)))
    ssim_matrix = np.zeros((len(methods), len(degradation_types)))

    # Total number of tasks for progress tracking
    total_tasks = len(methods) * len(degradation_types) * 200
    print(total_tasks, len(methods))
    # Create a progress bar
    with tqdm(total=total_tasks, desc="Processing Images", unit="task") as pbar:
        # Loop over methods
        for k, method in enumerate(methods):
            print(f"Processing method: {method}")
            
            # Loop over degradation types
            for j, degradation_type in enumerate(degradation_types):
                psnr_values = []
                ssim_values = []
                
                # Loop over each image in the clear folder
                for img_name in img_list:
                    clear_img_path = os.path.join(clear_folder, img_name)
                    degraded_img_path = f'./{method}/{degradation_type}/{img_name}'
                    degraded_img_path = os.path.join(degradation_path, degradation_type+'_'+img_name) 
                    # Read the clear and degraded images
                    clear_img = cv2.imread(clear_img_path) / 255.0
                    degraded_img = cv2.imread(degraded_img_path) / 255.0
                    
                    # Ensure the images are read correctly
                    if clear_img is not None and degraded_img is not None:
                        # Compute PSNR and SSIM between clear and degraded image
                        psnr_value = compare_psnr(clear_img, degraded_img, data_range=1.0)
                        
                        # Compute SSIM with specified window size and for multichannel images
                        ssim_value = compare_ssim(clear_img, degraded_img, multichannel=True, 
                                                  win_size=min(win_size, clear_img.shape[0], clear_img.shape[1]), 
                                                  channel_axis=-1, data_range=1.0)
                        
                        # Store values
                        psnr_values.append(psnr_value)
                        ssim_values.append(ssim_value)

                    # Update progress bar after processing each image
                    pbar.update(1)
                
                # Calculate mean PSNR and SSIM for the current method and degradation type
                if psnr_values:
                    psnr_matrix[k, j] = np.mean(psnr_values)
                if ssim_values:
                    ssim_matrix[k, j] = np.mean(ssim_values)

    return psnr_matrix, ssim_matrix

def save_matrices_to_excel(psnr_matrix, ssim_matrix, methods, degradation_types, data):
    # Create DataFrames for PSNR and SSIM matrices
    psnr_df = pd.DataFrame(psnr_matrix, index=methods, columns=degradation_types)
    ssim_df = pd.DataFrame(ssim_matrix, index=methods, columns=degradation_types)
    
    output_file = data + '.xlsx'
    # Create a writer to write both DataFrames to the same Excel file
    with pd.ExcelWriter(output_file) as writer:
        psnr_df.to_excel(writer, sheet_name='PSNR')
        ssim_df.to_excel(writer, sheet_name='SSIM')
    
    print(f'Matrices saved to {output_file}')



parser = argparse.ArgumentParser(description='Evaluation')

parser.add_argument('--data', default='CDD', choices=['CDD', 'CDD-Base'],
                    type=str, help='dataset')               

args = parser.parse_args()

# Define the parameters
opt = parse('./options/CDD.yml', is_train=False)
clear_folder = opt["datasets"]["val"]['dataroot_gt']
degradation_types = ['low', 'haze', 'rain', 'snow', 'low_haze', 'low_rain', 'low_snow', 'haze_rain', 'haze_snow', 'low_haze_rain', 'low_haze_snow']

degradation_path = os.path.join('./results/', args.data)
methods = ['BioIR']
# This is the function that will be used to calculate the PSNR and SSIM values across methods and degradation types
# To use the function, uncomment the line below and ensure the file paths are set correctly in your environment


psnr_matrix, ssim_matrix = calculate_psnr_ssim_with_progress(clear_folder, degradation_types, methods, degradation_path)
print(psnr_matrix, ssim_matrix)
save_matrices_to_excel(psnr_matrix, ssim_matrix, methods, degradation_types, args.data)
