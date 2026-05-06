# PAMP (Pellet and Mycelium Analysis Pipeline)

## Project Overview

PAMP is an image processing and analysis pipeline for analyzing morphological characteristics of microbial pellets and mycelium. This tool can automatically process images, extract key morphological parameters, and generate detailed analysis reports.

## Features

- **Pellet Analysis**: Analyze pellet core radius, peripheral hyphal length, projected area, circularity, porosity, etc.
- **Mycelium Analysis**: Analyze mycelium length, tip count, branch point count, etc.
- **Spore Analysis**: Analyze spore germ tube length, tip count, etc.
- **Automated Processing**: Support batch processing of multiple images
- **Result Visualization**: Generate multiple visualization images for easy result analysis
- **Result Export**: Export analysis results to CSV format for further analysis

## Project Structure

```
PAMP/
├── Hyphae.py         # Mycelium analysis module
├── Pellets.py        # Pellet analysis module
├── PAMP.py           # Main script, integrating both analysis modules
├── README.md         # Project description file
└── Docs.md           # This document
```

## Installation Requirements

### Dependencies

- Python 3.6+
- OpenCV (cv2)
- NumPy
- Matplotlib
- SciPy
- scikit-image
- tqdm

### Installation Command

```bash
pip install opencv-python numpy matplotlib scipy scikit-image tqdm
```

## Usage

### 1. Image Segmentation (Spore and Mycelium Images)

For spore and mycelium images, you need to first use the locally deployed Segment Anything tool for segmentation:

1. Start the locally deployed Segment Anything service
2. Upload spore or mycelium images
3. Use the tool for manual or automatic segmentation, ensuring the target area is correctly segmented
4. Download the segmented images

### 2. Prepare Input Data

Create the input directory structure with the following subdirectories:

```
input/
├── Pellets/    # Store pellet images
├── Hyphae/     # Store segmented mycelium images
└── Spores/     # Store segmented spore images
```

### 3. Run Analysis

Use the `PAMP.py` script to run the analysis:

```bash
python PAMP.py --input_folder ./input --output_folder ./output
```

### Command Line Parameters

| Parameter | Description | Default Value |
|-----------|-------------|---------------|
| `--input_folder` | Input folder path | `./Input` |
| `--output_folder` | Output folder path | `./Output` |
| `--Flat_field_img` | Flat field correction image path (optional) | `None` |
| `--pellet_scale` | Pixel scale factor for pellet analysis | `2.14774` |
| `--hyphae_scale` | Pixel scale factor for mycelium analysis | `0.3373` |

## Analysis Flow

### Pellet Analysis Flow

1. **Image Preprocessing**: Read the image and convert to grayscale
2. **Flat Field Correction**: (Optional) Use reference image for flat field correction
3. **Binarization**: Use Otsu threshold and adaptive threshold for binarization
4. **Morphological Operations**: Use morphological closing operation to fill holes
5. **Contour Extraction**: Extract outer and core contours of the pellet
6. **Parameter Calculation**: Calculate core radius, peripheral hyphal length, projected area, etc.
7. **Result Saving**: Save analysis results and visualization images

### Mycelium Analysis Flow

1. **Image Preprocessing**: Gaussian blur and CLAHE enhancement
2. **Binarization**: Use global threshold for binarization
3. **Morphological Operations**: Use morphological closing and opening operations
4. **Skeletonization**: Perform skeletonization on the binary image
5. **Skeleton Filtering**: Filter short branches and noise
6. **Parameter Calculation**: Calculate skeleton length, tip count, branch point count, etc.
7. **Result Saving**: Save analysis results and visualization images

## Output Results

### Pellet Analysis Output

- **CSV File**: `output/Pellets/Pellets results.csv`, containing the following parameters:
  - Pellet core radius (Rc)
  - Pellet core area (Ac)
  - Peripheral hyphal length (Lph)
  - Peripheral area (Aph)
  - Pellet radius (Rp)
  - Pellet projected area (Ap)
  - Hypothetical convex area (Ahc)
  - Aspect Ratio (AR)
  - Pellet Circularity (PC)
  - Pellet Solidity (PS)
  - Morphological Number (MN)
  - Pellet Porosity (PP)

- **Visualization Images**: Analysis result images for each pellet, including:
  - Original image
  - Binary image
  - Closing operation image
  - Contour image
  - Fitted ellipse image
  - Peripheral mask
  - Core mask
  - Outer mask
  - Merged mask color image

### Mycelium Analysis Output

- **CSV Files**:
  - `output/Spores/Spores results.csv`: Spore analysis results
  - `output/Hyphae/Hyphae results.csv`: Mycelium analysis results

- **Visualization Images**: Analysis result images for each sample, including:
  - Original image
  - Binary image
  - Skeleton image
  - Colored skeleton image
  - Merged binary and colored skeleton image

## Core Function Modules

### HyphaeAnalyzer Class

- **load_image**: Load image
- **preprocess**: Image preprocessing (Gaussian blur and CLAHE enhancement)
- **apply_global_threshold**: Apply global threshold
- **apply_morphology**: Apply morphological operations
- **perform_skeletonization**: Perform skeletonization
- **_filter_skeleton_noise**: Filter skeleton noise
- **color_skeleton_by_connectivity**: Color skeleton by connectivity
- **detect_and_mark_blue_clusters**: Detect and mark blue clusters (branch points)
- **count_red_pixels**: Count red pixels (skeleton main trunk)
- **count_green_pixels**: Count green pixels (mycelium tips)
- **analyze_hyphal_tip_and_branch_points**: Analyze hyphal tips and branch points
- **segment_individual_objects**: Segment individual objects
- **calculate_hyphae_area**: Calculate hyphae area
- **calculate_skeleton_length**: Calculate skeleton length
- **process_single_image**: Process single image
- **process_directory**: Process all images in a directory
- **generate_summary_csv**: Generate summary CSV file
- **process_multiple_directories**: Process multiple directories
- **calculate_circularity**: Calculate circularity

### Pellet Analysis Functions

- **calculate_circularity**: Calculate circularity
- **calculate_average_peripheral_max_distances**: Calculate average peripheral maximum distances
- **calc_peripheral_area_from_peripheral_mask**: Calculate peripheral area from peripheral mask
- **calculate_fitted_convex_perimeter**: Calculate fitted convex perimeter
- **calculate_diameter_projections**: Calculate diameter projections
- **calculate_average_core_max_distances**: Calculate average core maximum distances
- **process_image**: Process single image
- **save_images**: Save images
- **save_results_to_csv**: Save results to CSV file
- **pellet_analysis**: Perform pellet analysis

## Example Usage

### Analyze Single Directory

```bash
python PAMP.py --input_folder ./input --output_folder ./output
```

### Use Flat Field Correction

```bash
python PAMP.py --input_folder ./input --output_folder ./output --Flat_field_img ./flat_field.tif
```

### Adjust Scale Factors

```bash
python PAMP.py --input_folder ./input --output_folder ./output --pellet_scale 2.0 --hyphae_scale 0.3
```

## Notes

1. **Image Formats**: Supported image formats include PNG, JPG, JPEG, BMP, TIF, TIFF
2. **Input Directory Structure**: Input images must be organized according to the specified directory structure
3. **Computation Time**: Processing a large number of images may take a long time, please be patient
4. **Memory Usage**: Processing high-resolution images may require large memory
5. **Result Interpretation**: Analysis results should be interpreted in conjunction with actual experimental conditions

## Troubleshooting

1. **Image Loading Failure**: Check if the image path is correct and the image format is supported
2. **Memory Error**: Try processing smaller images or reduce the number of images processed in batch
3. **CSV File Writing Error**: Check write permissions for the output directory
4. **Abnormal Analysis Results**: Check input image quality, threshold parameters may need adjustment

## Version History

- **v1.0**: Initial version, including basic pellet and mycelium analysis functions

## Contribution

Welcome to submit issues and improvement suggestions!

## License

This project is licensed under the MIT License.