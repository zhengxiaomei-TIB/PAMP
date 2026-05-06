from tkinter import Scale
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import csv
from tqdm import tqdm
import json


def calculate_circularity(area, perimeter):
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)


def calculate_average_peripheral_max_distances(peripheral_mask, angle=0.1):
    points = np.column_stack(np.where(peripheral_mask > 0))[:, ::-1]
    
    if points.shape[0] == 0:
        return 0, np.array([])

    center = points.mean(axis=0)

    vecs = points - center
    angles = (np.degrees(np.arctan2(vecs[:,1], vecs[:,0])) + 360) % 360

    angle_bins = np.floor(angles / angle).astype(int)

    num_bins = int(360 / angle)
    
    max_distances = []
    max_distance_points = []
    for bin_idx in range(num_bins):

        bin_points = points[angle_bins == bin_idx]
        if len(bin_points) < 1:
            continue
        
        distances = np.linalg.norm(bin_points - center, axis=1)
        
        max_distance = np.max(distances)
        max_distances.append(max_distance)

        max_idx = np.argmax(distances)

        max_distance_point = bin_points[max_idx]
        max_distance_points.append(max_distance_point)
    
    if len(max_distances) == 0:
        return 0, np.array([])

    return max_distance, np.array(max_distance_points)

def calc_peripheral_area_from_peripheral_mask(peripheral_mask, angle=10, threshold_num = 100):

    max_distance_value, max_distance_points = calculate_average_peripheral_max_distances(peripheral_mask)
    points = max_distance_points
    
    if points.shape[0] == 0:
        return 0, None, 0

    center = points.mean(axis=0)

    vecs = points - center
    angles = (np.degrees(np.arctan2(vecs[:,1], vecs[:,0])) + 360) % 360

    angle_bins = np.floor(angles / angle).astype(int)

    num_bins = int(360 / angle)

    distances = np.linalg.norm(points - center, axis=1)
    max_distances_mean_dist = np.mean(distances)
    max_distances_std_dist = np.std(distances)

    pellet_radius_piexl = max_distances_mean_dist
    
    filtered_points = []
    for bin_idx in range(num_bins):

        bin_points = points[angle_bins == bin_idx]
        if len(bin_points) < 2:
            continue

        distances = np.linalg.norm(bin_points - center, axis=1)

        mean_dist = max_distances_mean_dist
        std_dist = max_distances_std_dist

        threshold_num = 100

        threshold = min(mean_dist + threshold_num * std_dist, np.max(distances))
        valid_indices = np.where(distances <= threshold)[0]

        if len(valid_indices) == 0:
            valid_indices = np.arange(len(bin_points))
        
        if len(valid_indices) > 0:
            filtered_points.extend(bin_points[valid_indices])
    
    filtered_points = np.array(filtered_points)
    if len(filtered_points) < 5:
        if len(points) >= 5:
            filtered_points = points
        else:
            return 0, None, 0
    
    fitted_convex = cv2.fitEllipse(filtered_points.astype(np.int32))

    (xc, yc), (d1, d2), fitted_convex_angle = fitted_convex
    
    min_fitted_convex_area = np.pi * 10 * 10
    fitted_convex_area = np.pi * (d1 / 2) * (d2 / 2)

    return fitted_convex_area, fitted_convex, pellet_radius_piexl

def calculate_fitted_convex_perimeter(a, b):
    h = ((a - b) ** 2) / ((a + b) ** 2)
    perimeter = np.pi * (a + b) * (1 + (3 * h) / (10 + np.sqrt(4 - 3 * h)))
    return perimeter


def calculate_diameter_projections(contour):
    rect = cv2.minAreaRect(contour)
    (cx, cy), (width, height), angle = rect
    longest_projection = max(width, height)
    shortest_projection = min(width, height)
    return {
        'longest_projection': longest_projection,
        'shortest_projection': shortest_projection
    }



def calculate_average_core_max_distances(core_mask, angle=0.1):

    points = np.column_stack(np.where(core_mask > 0))[:, ::-1]
    
    if points.shape[0] == 0:
        return 0, np.array([])

    center = points.mean(axis=0)

    vecs = points - center
    angles = (np.degrees(np.arctan2(vecs[:,1], vecs[:,0])) + 360) % 360

    angle_bins = np.floor(angles / angle).astype(int)

    num_bins = int(360 / angle)
    
    max_distances = []
    max_distance_points = []
    for bin_idx in range(num_bins):

        bin_points = points[angle_bins == bin_idx]
        if len(bin_points) < 1:
            continue
        
        distances = np.linalg.norm(bin_points - center, axis=1)
        
        max_distance = np.max(distances)
        max_distances.append(max_distance)
        
        max_idx = np.argmax(distances)
        
        max_distance_point = bin_points[max_idx]
        max_distance_points.append(max_distance_point)
    
    if len(max_distances) == 0:
        return 0, np.array([])
    
    core_average_radius_pixel = np.mean(max_distances)
    
    return core_average_radius_pixel, np.array(max_distance_points)

def process_image(image_path, Flat_field_img=None, flat_field_normalized=None, scale=0.793398):

    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if Flat_field_img is not None:
        gray = (cv2.min(gray / flat_field_normalized, 255)).astype(np.uint8)

    gray = 255 - gray

    _, origin_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_edge = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, -10)

    initial_core_binary = np.bitwise_and(origin_binary, 255 - binary_edge)
    finetune_core_binary = cv2.adaptiveThreshold(initial_core_binary, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, -10)
    core_binary = np.bitwise_and(initial_core_binary, finetune_core_binary)

    binary = cv2.bitwise_or(origin_binary, binary_edge)

    kernel_size = 11
    kernel = np.zeros((kernel_size, kernel_size), np.uint8)
    cv2.circle(kernel, (kernel_size // 2, kernel_size // 2), kernel_size // 2, 1, -1)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=5)

    outer_contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_outer_contour = max(outer_contours, key=cv2.contourArea)
    outer_hull = cv2.convexHull(largest_outer_contour)

    core_contours, _ = cv2.findContours(core_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_core_contour = max(core_contours, key=cv2.contourArea)
    core_hull = cv2.convexHull(largest_core_contour)

    core_mask = np.zeros_like(binary)
    cv2.drawContours(core_mask, [largest_core_contour], -1, 255, -1)

    temp_outer_mask = np.zeros_like(binary)
    cv2.drawContours(temp_outer_mask, [outer_hull], -1, 255, -1)

    peripheral_mask = np.bitwise_and(np.bitwise_and(binary_edge, temp_outer_mask), 255 - core_mask)

    outer_area = cv2.contourArea(outer_hull)
    core_projected_area = cv2.contourArea(largest_core_contour)
    core_projected_area_scaled = core_projected_area * (scale ** 2)

    peripheral_projected_area = peripheral_mask.sum() / 255
    peripheral_projected_area_scaled = peripheral_projected_area * (scale ** 2)

    core_average_radius_pixel, _ = calculate_average_core_max_distances(core_mask)
    core_average_radius_pixel_scaled = core_average_radius_pixel * scale

    outer_area_fitted_convex, outer_fitted_convex, pellet_radius_pixel = calc_peripheral_area_from_peripheral_mask(peripheral_mask)
    pellet_radius_pixel_scaled = pellet_radius_pixel * scale

    fitted_convex_img = image.copy()
    outer_mask = None
    if outer_fitted_convex is None:

        outer_mask = temp_outer_mask
        major_axis = 0
        minor_axis = 0
        fitted_hypothetical_convex_perimeter = 0
        outer_circularity = 0
        fitted_hypothetical_convex_area = 0
    else:

        outer_mask = np.zeros_like(binary)
        cv2.ellipse(outer_mask, outer_fitted_convex, 255, -1)
        cv2.ellipse(fitted_convex_img, outer_fitted_convex, (0, 255, 0), 2)
        (xc, yc), (d1, d2), angle = outer_fitted_convex
        major_axis = max(d1, d2)
        minor_axis = min(d1, d2)
        fitted_hypothetical_convex_perimeter = calculate_fitted_convex_perimeter(major_axis / 2, minor_axis / 2)
        outer_circularity = calculate_circularity(outer_area_fitted_convex, fitted_hypothetical_convex_perimeter)
        fitted_hypothetical_convex_area = np.pi * (major_axis / 2) * (minor_axis / 2) * (scale ** 2)

    core_circle_radius = np.sqrt(core_projected_area / np.pi)
    peripheral_hypha_length_pixel = max(0, pellet_radius_pixel - core_circle_radius)
    peripheral_hypha_length_um = peripheral_hypha_length_pixel * scale

    core_diameter = calculate_diameter_projections(largest_core_contour)
    core_aspect_ratio = (core_diameter['longest_projection'] / core_diameter['shortest_projection']) if core_diameter['shortest_projection'] > 0 else 0

    perimeter_core_hull = cv2.arcLength(core_hull, True)
    core_hull_circular = calculate_circularity(core_projected_area, perimeter_core_hull)

    total_projected_area = core_projected_area_scaled + peripheral_projected_area_scaled
    solidity = core_projected_area / max(core_projected_area + peripheral_projected_area, 1e-9)

    denom_area = max(fitted_hypothetical_convex_area - core_projected_area_scaled, 1e-9)
    porosity = 1 - (peripheral_projected_area_scaled / denom_area)

    pellet_aspect_ratio = (major_axis / minor_axis) if minor_axis > 0 else 0
    if pellet_aspect_ratio > 0 and pellet_radius_pixel_scaled > 0:
        MN = (2 * np.sqrt(total_projected_area) * solidity) / (np.sqrt(np.pi) * pellet_radius_pixel_scaled*2 * pellet_aspect_ratio)
    else:
        MN = 0

    result = {
        'Sample name': os.path.basename(image_path),
        'Pellet core radius (Rc)': core_average_radius_pixel_scaled,
        'Pelletcore area (Ac)': core_projected_area_scaled,
        'Peripheral hyphal length (Lph)': peripheral_hypha_length_um,
        'Peripheral area (Aph)': peripheral_projected_area_scaled,
        'Pellet radius (Rp)': pellet_radius_pixel_scaled,
        'Pellet projected area (Ap)': total_projected_area,
        'Hypothetical convex area (Ahc)': fitted_hypothetical_convex_area,
        'Aspect Ratio (AR)': pellet_aspect_ratio,
        'Pellet Circularity (PC)': outer_circularity if outer_fitted_convex is not None else 0,
        'Pellet Solidity (PS)': solidity,
        'Morphological Number (MN)': MN,
        'Pellet Porosity (PP)': porosity
    }

    contour_img = image.copy()
    if outer_fitted_convex is not None:
        cv2.ellipse(contour_img, outer_fitted_convex, (0, 255, 0), 3)
    cv2.drawContours(contour_img, [largest_core_contour], -1, (0, 0, 255), 3)
    cv2.drawContours(contour_img, [outer_hull], -1, (0, 255, 255), 3)

    merged_mask = np.zeros_like(peripheral_mask)
    merged_mask[core_mask > 0] = 255
    merged_mask[peripheral_mask > 0] = 255
    merged_mask_color = cv2.cvtColor(merged_mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(merged_mask_color, [largest_core_contour], -1, (0, 0, 255), 2)
    if outer_fitted_convex is not None:
        cv2.ellipse(merged_mask_color, outer_fitted_convex, (0, 255, 0), 2)

    images_dict = {
        'image': image,
        'binary': binary,
        'closing': closing,
        'contour_img': contour_img,
        'fitted_convex_img': fitted_convex_img,
        'peripheral_mask': peripheral_mask,
        'core_mask': core_mask,
        'outer_mask': outer_mask if outer_mask is not None else temp_outer_mask,
        'merged_mask_color': merged_mask_color
    }

    return result, images_dict


def save_images(output_folder, index, img_name, images_dict):

    slice_folder = os.path.join(output_folder, img_name)
    if not os.path.exists(slice_folder):
        os.makedirs(slice_folder)

    for name, img in images_dict.items():
        plt.figure(num=name)
        plt.imshow(img[..., ::-1] if img.shape[-1] == 3 else img, cmap='gray' if name in ['peripheral_mask', 'core_mask', 'outer_mask', 'binary', 'closing'] else None)
        plt.axis('off')  
        plt.savefig(os.path.join(slice_folder, f'{index+1}-{name}.png'), bbox_inches='tight', pad_inches=0)
        plt.close()

def save_results_to_csv(output_folder, results):

    os.makedirs(output_folder, exist_ok=True)

    file_name = os.path.join(output_folder, 'Pellets results.csv')
    try:
        with open(file_name, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['index'] + list(results[0].keys()))
            for i, data in enumerate(results):
                csvwriter.writerow([i+1] + list(data.values()))
        print(f"Results successfully written to: {file_name}")
    except PermissionError:
        print(f"Error: No permission to write file {file_name}")

        alt_file_name = os.path.join(os.getcwd(), 'Pellets results.csv')
        with open(alt_file_name, 'w', newline='', encoding='utf-8') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['index'] + list(results[0].keys()))
            for i, data in enumerate(results):
                csvwriter.writerow([i+1] + list(data.values()))
        print(f"Results written to alternative location: {alt_file_name}")
    except Exception as e:
        print(f"Error occurred while writing file: {e}")


def pellet_analysis(input_folder, output_folder, Flat_field_img_path=None, scale=2.14774):

    Flat_field_img = None
    flat_field_normalized = None
    if Flat_field_img_path and Flat_field_img_path != "":
        Flat_field_img = cv2.imread(Flat_field_img_path, cv2.IMREAD_GRAYSCALE)
        Flat_field_img = cv2.GaussianBlur(Flat_field_img, (101, 101), 0)
        flat_field_normalized = Flat_field_img / np.max(Flat_field_img)

    # Only process the 'pellet' folder, not all folders
    group_folder = "Pellets"
    input_group_folder = os.path.join(input_folder, group_folder)
    if os.path.exists(input_group_folder) and os.path.isdir(input_group_folder):
        output_group_folder = os.path.join(output_folder, group_folder)
        if not os.path.exists(output_group_folder):
            os.makedirs(output_group_folder)

        results = []
        img_lists = os.listdir(input_group_folder)
        for index, img_name in tqdm(enumerate(img_lists), total=len(img_lists)):
            image_path = os.path.join(input_group_folder, img_name)
            result, images_dict = process_image(
                image_path,
                Flat_field_img=Flat_field_img,
                flat_field_normalized=flat_field_normalized,
                scale=scale
            )
            results.append(result)

            save_images(output_group_folder, index, img_name, images_dict)

        save_results_to_csv(output_group_folder, results)


def main():

    parser = argparse.ArgumentParser(description='Pellet processing')
    parser.add_argument('--scale', type=float, help='micrometers per pixel', default=2.14774)
    parser.add_argument('--input_folder', type=str, default='input', help='folder containing input images')
    parser.add_argument('--output_folder', type=str, default='./output', help='folder to save results')
    parser.add_argument('--Flat_field_img', type=str, default=None, help='path to reference image (optional, for flat-field correction)')
    
    args = parser.parse_args()

    pellet_analysis(
        args.input_folder,
        args.output_folder,
        Flat_field_img_path=args.Flat_field_img,
        scale=args.scale
    )


if __name__ == '__main__':
    main()