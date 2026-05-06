import cv2
import numpy as np
import sys
from scipy.ndimage import convolve
from skimage.morphology import skeletonize
import matplotlib.pyplot as plt
import os
import glob
import csv
from tqdm import tqdm
import json

class HyphaeAnalyzer:

    def __init__(self):
        self.gaussian_kernel = (5, 5)
        self.clahe_clip_limit = 0.8
        self.clahe_grid_size = (12, 12)
        self.global_threshold = 10
        self.adaptive_block_size = 11
        self.adaptive_C = 2
        self.morph_kernel_size = (3, 3)
        self.pixel_size = 0.3373
        self.min_object_size = 200
        self.results_summary = {}

    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        return img

    def preprocess(self, img):
        blurred = cv2.GaussianBlur(img, self.gaussian_kernel, 0)
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, 
                               tileGridSize=self.clahe_grid_size)
        clahe_enhanced = clahe.apply(blurred)
        
        return blurred, clahe_enhanced
    
    def apply_global_threshold(self, clahe_enhanced, threshold=None):

        if threshold is None:
            threshold = self.global_threshold
        _, binary = cv2.threshold(clahe_enhanced, threshold, 255, cv2.THRESH_BINARY)
        return binary            
    
    def apply_morphology(self, binary):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, self.morph_kernel_size)
        
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=3)
        
        return closed, opened
    
    def perform_skeletonization(self, binary_image):

        _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

        binary_bool = binary // 255
        skeleton = skeletonize(binary_bool)

        skeleton_uint8 = (skeleton * 255).astype(np.uint8)

        skeleton_filtered = self._filter_skeleton_noise(skeleton_uint8)
        
        return skeleton_filtered
    
    def _filter_skeleton_noise(self, skeleton, min_branch_length=5):

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(skeleton, connectivity=8)

        filtered_skeleton = np.zeros_like(skeleton)
        
        for i in range(1, num_labels): 
            component_area = stats[i, cv2.CC_STAT_AREA]
            if component_area >= min_branch_length:
                filtered_skeleton[labels == i] = 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        neighbors = cv2.filter2D(filtered_skeleton.astype(np.float32)/255, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        filtered_skeleton[(filtered_skeleton > 0) & (neighbors < 1)] = 0
        
        return filtered_skeleton
    
    def color_skeleton_by_connectivity(self, skeleton):

        h, w = skeleton.shape
        colored_skeleton = np.zeros((h, w, 3), dtype=np.uint8)

        GREEN = (0, 255, 0)   
        RED = (0, 0, 255)     
        BLUE = (255, 50, 50)  

        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),          (0, 1),
                     (1, -1),  (1, 0), (1, 1)]

        hyphal_tip = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(h):
            for j in range(w):
                if skeleton[i, j] == 255:
                    count = 0
                    for di, dj in neighbors:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            if skeleton[ni, nj] == 255:
                                count += 1

                    if count == 1:
                        hyphal_tip[i, j] = 255
                    elif count == 2:
                        colored_skeleton[i, j] = RED 
                    elif count >= 3:
                        colored_skeleton[i, j] = BLUE 

        if np.any(hyphal_tip):
            kernel = np.ones((3, 3), np.uint8) 
            expanded_hyphal_tip = cv2.dilate(hyphal_tip, kernel, iterations=1)
            colored_skeleton[expanded_hyphal_tip == 255] = GREEN

        skeleton_mask = np.zeros((h, w), dtype=np.uint8)
        skeleton_mask[skeleton == 255] = 255

        thick_kernel = np.ones((5, 5), np.uint8)
        thickened_skeleton_mask = cv2.dilate(skeleton_mask, thick_kernel, iterations=2)

        thickened_colored_skeleton = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(h):
            for j in range(w):
                if thickened_skeleton_mask[i, j] == 255:
                    if np.any(colored_skeleton[i, j] != 0):
                        thickened_colored_skeleton[i, j] = colored_skeleton[i, j]
                    else:
                        for di, dj in neighbors:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < h and 0 <= nj < w and np.any(colored_skeleton[ni, nj] != 0):
                                thickened_colored_skeleton[i, j] = colored_skeleton[ni, nj]
                                break
        return thickened_colored_skeleton

    def detect_and_mark_blue_clusters(self, colored_skeleton):

        marked_image = colored_skeleton.copy()

        blue_mask = np.zeros(colored_skeleton.shape[:2], dtype=np.uint8)
        blue_mask[(colored_skeleton[:, :, 0] > 200) & (colored_skeleton[:, :, 1] < 100) & (colored_skeleton[:, :, 2] < 100)] = 255

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)

        yellow_circles_centers = []

        for i in range(1, num_labels):
            x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]

            center_x = x + w // 2
            center_y = y + h // 2
            radius = int(max(w, h) * 0.7)

            cv2.circle(marked_image, (center_x, center_y), radius + 2, (0, 255, 255), 3)

            yellow_circles_centers.append((center_x, center_y))

        return marked_image, len(yellow_circles_centers)
    
    def count_red_pixels(self, colored_skeleton):

        red_mask = np.all(colored_skeleton == (0, 0, 255), axis=-1)
        red_pixel_count = np.sum(red_mask)
        
        return red_pixel_count
    
    def count_green_pixels(self, colored_skeleton):

        green_mask = np.all(colored_skeleton == (0, 255, 0), axis=-1)
        green_pixel_count = np.sum(green_mask)
        
        return green_pixel_count
    
    def analyze_hyphal_tip_and_branch_points(self, colored_skeleton, sample_type):

        green_pixel_count = self.count_green_pixels(colored_skeleton)

        green_mask = np.all(colored_skeleton == (0, 255, 0), axis=-1)

        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(green_mask.astype(np.uint8), connectivity=8)

        if sample_type == 'Spores':
            actual_hyphal_tip_count = num_labels - 2
            hyphal_tip_count = actual_hyphal_tip_count
        else:
            actual_hyphal_tip_count = num_labels - 1
            hyphal_tip_count = actual_hyphal_tip_count

        marked_image, yellow_circles_count = self.detect_and_mark_blue_clusters(colored_skeleton)

        branch_points_count = yellow_circles_count
        
        return hyphal_tip_count, branch_points_count

    def segment_individual_objects(self, binary_image, min_size=10):
        _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        object_masks = []
        valid_objects_count = 0

        for i in range(1, num_labels):

            if stats[i, cv2.CC_STAT_AREA] >= min_size:
                object_mask = np.zeros_like(binary_image)
                object_mask[labels == i] = 255
                object_masks.append(object_mask)
                valid_objects_count += 1
        
        return object_masks, valid_objects_count
        
    def calculate_hyphae_area(self, binary_image, skeleton=None, sample_type=None):

        _, binary = cv2.threshold(binary_image, 127, 255, cv2.THRESH_BINARY)

        area = np.sum(binary == 255)

        area_actual = None
        if self.pixel_size is not None:
            area_actual = area * (self.pixel_size ** 2)
        
        return area, area_actual

    def calculate_skeleton_length(self, skeleton):

        skeleton_points = np.column_stack(np.where(skeleton == 255))

        if len(skeleton_points) == 0:
            return 0, 0 if self.pixel_size is not None else None

        length = 0.0

        point_set = set(map(tuple, skeleton_points))
        visited = set()

        directions = [
            (0, 1, 1.0), (1, 0, 1.0), (0, -1, 1.0), (-1, 0, 1.0),
            (1, 1, np.sqrt(2)), (1, -1, np.sqrt(2)),
            (-1, 1, np.sqrt(2)), (-1, -1, np.sqrt(2))
        ]

        # Function to count neighbors
        def count_neighbors(y, x):
            count = 0
            for dy, dx, _ in directions:
                ny, nx = y + dy, x + dx
                if (ny, nx) in point_set:
                    count += 1
            return count

        # Find tip points (points with only one neighbor)
        tip_points = []
        for point in skeleton_points:
            y, x = point
            if count_neighbors(y, x) == 1:
                tip_points.append((y, x))

        # If no tips found, use all points as starting points
        if not tip_points:
            tip_points = [tuple(point) for point in skeleton_points]

        # Traverse each tip to calculate length
        for start_point in tip_points:
            if start_point in visited:
                continue

            current = start_point
            prev = None

            while True:
                if current in visited:
                    break

                visited.add(current)

                # Find next neighbors
                neighbors = []
                for dy, dx, dist in directions:
                    ny, nx = current[0] + dy, current[1] + dx
                    neighbor = (ny, nx)
                    if neighbor in point_set and neighbor != prev:
                        neighbors.append((neighbor, dist))

                if not neighbors:
                    break

                # For multiple neighbors, this is a branch point
                # We'll handle each branch separately
                for neighbor, dist in neighbors:
                    if neighbor not in visited:
                        length += dist
                        prev = current
                        current = neighbor
                        break
                else:
                    break

        length_actual = None
        if self.pixel_size is not None:
            length_actual = length * self.pixel_size

        return length, length_actual

    def process_single_image(self, image_path, output_dir=None, sample_type='Hyphae'):
        try:
            img = self.load_image(image_path)

            blurred, clahe_enhanced = self.preprocess(img)

            binary = self.apply_global_threshold(clahe_enhanced)

            closed, opened = self.apply_morphology(binary)

            skeleton = self.perform_skeletonization(opened)

            colored_skeleton = self.color_skeleton_by_connectivity(skeleton)

            marked_image, _ = self.detect_and_mark_blue_clusters(colored_skeleton)

            hyphal_tip_count, branch_points_count = self.analyze_hyphal_tip_and_branch_points(colored_skeleton, sample_type)

            object_analyses = []

            if sample_type == 'Spores':
                object_masks, num_objects = self.segment_individual_objects(opened, min_size=self.min_object_size)
                print(f"Detected {num_objects} spore objects in image {os.path.basename(image_path)}")

                valid_objects = []
                filtered_skeleton = np.zeros_like(skeleton) 

                for i, object_mask in enumerate(object_masks):
                    _, object_binary = cv2.threshold(object_mask, 127, 255, cv2.THRESH_BINARY)
                    object_skeleton = self.perform_skeletonization(object_binary)
                    object_length, object_length_actual = self.calculate_skeleton_length(object_skeleton)

                    connected_pixels_count = np.sum(object_binary == 255)

                    if (object_length_actual is not None and object_length_actual >= 10 and 
                        connected_pixels_count > 100):
                        valid_objects.append((i, object_mask, object_binary, object_skeleton))
                        filtered_skeleton[object_skeleton > 0] = 255

                colored_skeleton = self.color_skeleton_by_connectivity(filtered_skeleton)
                marked_image, _ = self.detect_and_mark_blue_clusters(colored_skeleton)
                hyphal_tip_count, branch_points_count = self.analyze_hyphal_tip_and_branch_points(colored_skeleton, sample_type)
            else:
                valid_objects = []

            binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            merged_image = binary_color.copy()
            
            skeleton_mask = cv2.cvtColor(marked_image, cv2.COLOR_BGR2GRAY)
            skeleton_mask = cv2.threshold(skeleton_mask, 1, 255, cv2.THRESH_BINARY)[1]
            merged_image[skeleton_mask > 0] = marked_image[skeleton_mask > 0]

            if sample_type == 'Spores':
                h, w = img.shape[:2]
                
                for new_id, (original_id, object_mask, object_binary, object_skeleton) in enumerate(valid_objects, 1):
                    contours, _ = cv2.findContours(object_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours:
                        main_contour = max(contours, key=cv2.contourArea)

                        area = cv2.contourArea(main_contour)
                        perimeter = cv2.arcLength(main_contour, True)
                    
                    if perimeter > 0:
                        hull = cv2.convexHull(main_contour)

                        hull_area = cv2.contourArea(hull)
                        hull_perimeter = cv2.arcLength(hull, True)

                        hull_circularity = self.calculate_circularity(hull_area, hull_perimeter)

                        if hull_circularity > 0.5: 
                            M = cv2.moments(hull)
                        else:
                            M = cv2.moments(main_contour)
                        
                        if M['m00'] > 0:
                            cX = int(M['m10'] / M['m00'])
                            cY = int(M['m01'] / M['m00'])

                            min_y_point = tuple(main_contour[main_contour[:, :, 1].argmin()][0])
                            
                            angle = 0  
                            if len(hull) >= 5:
                                rect = cv2.minAreaRect(hull)
                                angle = rect[2]

                                if angle < -45:
                                    angle += 90

                                offset_distance = 20
                                offset_x = int(offset_distance * np.sin(np.radians(angle)))
                                offset_y = int(-offset_distance * np.cos(np.radians(angle)))
                                
                                display_X = min_y_point[0] + offset_x
                                display_Y = min_y_point[1] + offset_y
                            else:
                                display_X = min_y_point[0]
                                display_Y = min_y_point[1] - 20 

                            display_X = max(10, min(display_X, w - 10))
                            display_Y = max(20, min(display_Y, h - 10))

                            cv2.putText(merged_image, str(new_id), (display_X, display_Y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        object_colored_skeleton = self.color_skeleton_by_connectivity(object_skeleton)

                        _, object_blue_cluster_count = self.detect_and_mark_blue_clusters(object_colored_skeleton)

                        object_hyphal_tip_count, object_branch_points_count = self.analyze_hyphal_tip_and_branch_points(
                            object_colored_skeleton, sample_type
                        )

                        object_area, object_area_actual = self.calculate_hyphae_area(object_binary, object_skeleton, sample_type)

                        object_length, object_length_actual = self.calculate_skeleton_length(object_skeleton)

                        object_analyses.append({
                            'index': new_id,
                            'hyphal_tip_count': object_hyphal_tip_count,
                            'branching_points': object_branch_points_count,
                            'red_pixels': self.count_red_pixels(object_colored_skeleton),
                            'blue_clusters': object_blue_cluster_count,
                            'area': object_area,
                            'area_actual': object_area_actual,
                            'length': object_length,
                            'length_actual': object_length_actual
                        })
                    
                print(f"Filtered out {len(valid_objects)} spore objects in image {os.path.basename(image_path)} (length >= 10 units and connected pixels > 100)")
            elif sample_type == 'hyphae':
                pass

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            if output_dir is None:
                output_dir = "output_results"

            os.makedirs(output_dir, exist_ok=True)

            image_subdir = os.path.join(output_dir, base_name)
            os.makedirs(image_subdir, exist_ok=True)

            img_path = os.path.join(image_subdir, f"{base_name}_original.jpg")
            cv2.imwrite(img_path, img)
            
            binary_path = os.path.join(image_subdir, f"{base_name}_binary.jpg")
            cv2.imwrite(binary_path, binary)
            
            skeleton_path = os.path.join(image_subdir, f"{base_name}_skeleton.jpg")
            cv2.imwrite(skeleton_path, skeleton)
            
            marked_image_path = os.path.join(image_subdir, f"{base_name}_colored_skeleton.jpg")
            cv2.imwrite(marked_image_path, marked_image)

            merged_image_path = os.path.join(image_subdir, f"{base_name}_merged_binary_colored_skeleton.jpg")
            cv2.imwrite(merged_image_path, merged_image)

            print(f"Image {base_name} analysis completed, CSV file will be generated during batch processing")
            
            plt.close() 
            
            print(f"Image {image_path} processing completed")

            total_length, total_length_actual = self.calculate_skeleton_length(skeleton)
            
            length_key = 'germ_tube_length' if sample_type == 'Spores' else 'total_hyphal_length'
            length_actual_key = 'germ_tube_length (actual)' if sample_type == 'Spores' else 'total_hyphal_length (actual)'
            
            analysis_data = {
                'Sample name': base_name,
                'hyphal_tip_count': hyphal_tip_count,
                'branching_points': branch_points_count,
                'total_length': total_length,  
                'total_length_actual': total_length_actual,  
                length_key: total_length,  
                length_actual_key: total_length_actual, 
                'object_analyses': object_analyses  # 
            }
            
            return True, (img, binary, skeleton, colored_skeleton, merged_image), analysis_data
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return False, None, None

    def process_directory(self, directory_path, output_dir=None, sample_type='hyphae'):

        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']

        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(directory_path, ext)))
        
        if not image_paths:
            print(f"No supported image files found in directory: {directory_path}")
            return

        if output_dir is None:
            output_dir = os.path.join(directory_path, "batch_output_results")
        
        print(f"Found {len(image_paths)} image files, starting batch processing...")
        print(f"Results will be saved to: {output_dir}")

        if sample_type not in self.results_summary:
            self.results_summary[sample_type] = []

        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            result = self.process_single_image(image_path, output_dir, sample_type)
            if result is not None:
                success, images, analysis_data = result
                if success and analysis_data:
                    self.results_summary[sample_type].append(analysis_data)
            else:
                print(f"Warning: Processing of image {image_path} returned None")
        
        print("Batch processing completed")

    def generate_summary_csv(self, base_output_dir):
        for sample_type, results in self.results_summary.items():
            if not results:
                continue

            csv_dir = os.path.join(base_output_dir, sample_type)
            os.makedirs(csv_dir, exist_ok=True)
            csv_filename = f"{sample_type} results.csv"
            csv_path = os.path.join(csv_dir, csv_filename)

            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                if sample_type == 'Spores':
                    fieldnames = ['Sample name', 'Spore ID', 'Germ tube length (Lgs)', 'Hyphal tip count (HTgs)', 'Branching Points (BPgs)']
                else:  
                    fieldnames = ['Sample name', 'Total hyphal length (Lh)', 'Hyphal Tip Frequency (HTF)', 'Branching Points Frequency (BPF)']
                
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for result in results:
                    image_name = result.get('Sample name', '')
                    
                    if 'object_analyses' in result and result['object_analyses']:
                        for obj in result['object_analyses']:
                            length_actual = obj.get('length_actual', 0)
                            hyphal_tip_count = obj.get('hyphal_tip_count', 0)
                            branching_points = obj.get('branching_points', 0)
                            
                            if sample_type == 'Spores':
                                row_data = {
                                    'Sample name': image_name,
                                    'Spore ID': obj.get('index', ''),
                                    'Germ tube length (Lgs)': length_actual if length_actual is not None else '',
                                    'Hyphal tip count (HTgs)': hyphal_tip_count,
                                    'Branching Points (BPgs)': branching_points
                                }
                            else:  
                                if length_actual and length_actual > 0:
                                    tip_frequency = round(hyphal_tip_count / length_actual, 2)
                                    branch_frequency = round(branching_points / length_actual, 2)
                                else:
                                    tip_frequency = 0
                                    branch_frequency = 0
                                row_data = {
                                    'Sample name': image_name,
                                    'Total hyphal length (Lh)': length_actual if length_actual is not None else '',
                                    'Hyphal Tip Frequency (HTF)': tip_frequency,
                                    'Branching Points Frequency (BPF)': branch_frequency
                                }
                            writer.writerow(row_data)
                    else:
                        length_actual_total = result.get('total_length_actual', 0)
                        total_tips = result.get('hyphal_tip_count', 0)
                        total_branches = result.get('branching_points', 0)
                        
                        if sample_type == 'Spores':
                            total_row = {
                                'Sample name': image_name,
                                'Spore ID': 'total',
                                'Germ tube length (Lgs)': length_actual_total if length_actual_total is not None else '',
                                'Hyphal tip count (HTgs)': total_tips,
                                'Branching Points (BPgs)': total_branches
                            }
                        else:  
                            if length_actual_total and length_actual_total > 0:
                                total_tip_frequency = round(total_tips / length_actual_total, 2)
                                total_branch_frequency = round(total_branches / length_actual_total, 2)
                            else:
                                total_tip_frequency = 0
                                total_branch_frequency = 0
                            total_row = {
                                'Sample name': image_name,
                                'Total hyphal length (Lh)': length_actual_total if length_actual_total is not None else '',
                                'Hyphal Tip Frequency (HTF)': total_tip_frequency,
                                'Branching Points Frequency (BPF)': total_branch_frequency
                            }
                        writer.writerow(total_row)
            
            print(f"Generating summary CSV file (including object-level results) for {sample_type} type: {csv_path}")

    def process_multiple_directories(self, base_input_dir, sample_types, base_output_dir="batch_output_results"):
        if not os.path.exists(base_input_dir):
            print(f"Base input directory does not exist: {base_input_dir}")
            return

        self.results_summary = {}

        for sample_type in sample_types:
            input_dir = os.path.join(base_input_dir, sample_type)
            output_dir = os.path.join(base_output_dir, sample_type)

            if os.path.exists(input_dir) and os.path.isdir(input_dir):
                print(f"\nStarting processing for {sample_type} type samples, input directory: {input_dir}")
                self.process_directory(input_dir, output_dir, sample_type)
            else:
                print(f"Warning: {sample_type} subdirectory does not exist or is not a directory: {input_dir}")
        
        self.generate_summary_csv(base_output_dir)
        
        print(f"\nAll sample types processed, results saved in: {base_output_dir}")

    def calculate_circularity(self, area, perimeter):
        if perimeter == 0:
            return 0
        return (4 * np.pi * area) / (perimeter ** 2)

if __name__ == "__main__":

    hyphae_analyzer = HyphaeAnalyzer()

    hyphae_analyzer.pixel_size = 0.3373 
    
    process_mode = "multiple_directories" 
    
    if process_mode == "single_directory":
        input_folder_path = "images_to_process" 
        output_folder_path = "images_to_process_results"
        sample_type = "Spores" 

        hyphae_analyzer.process_directory(input_folder_path, output_folder_path, sample_type)
    elif process_mode == "multiple_directories":
        base_input_dir = "input" 
        sample_types = ['Spores', 'Hyphae']  

        hyphae_analyzer.process_multiple_directories(base_input_dir, sample_types)
