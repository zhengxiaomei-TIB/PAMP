# Image Analysis: Pellets and Hyphae

This project is for two types of microscopic image analysis:
- Pellet morphological analysis (`pellet.py`): Fits outlines, separates core from periphery, calculates radius, area, aspect ratio, circularity, porosity, MN, and other metrics, and outputs visualizations and CSV files.
- Hyphae network analysis (`hyphae.py`): Image preprocessing, segmentation, morphological processing, skeletonization, connectivity coloring (tips/branches/segments), statistics on tips and branches, total area and length (in pixels and actual units), spore mode supports filtering and numbering by object.

The command line entry point is `No_Note/main.py`, which can perform both analyses simultaneously in one run.

---

## Environment Dependencies
- Python ≥ 3.10 (3.11 recommended)
- Dependencies: `opencv-python`, `numpy`, `scipy`, `scikit-image`, `matplotlib`, `tqdm`

Installation example:
```bash
pip install opencv-python numpy scipy scikit-image matplotlib tqdm
```

---

## Directory Structure (Example)
```
No_Note/
  ├── main.py
  ├── hyphae.py
  ├── pellet.py
  └── README.md
input/
  ├── pellet/           # Pellet group data (one subdirectory per group, containing images)
  ├── hyphae/           # Hyphae data (optional)
  └── spore/            # Spore data (optional)
output/                 # Generated after running
```

---

## Quick Start
1) Prepare data directories: Place images in `input/pellet/` (subdirectories by group) and optional `input/hyphae/`, `input/spore/`.
2) Run the command:
```bash
# Windows (in repository root)
python No_Note\main.py --input_folder ./input --output_folder ./output \
  --pellet_scale 2.14774 --hyphae_scale 0.3373 \
  --reference_img path\to\reference.png

# Linux/macOS
python No_Note/main.py --input_folder ./input --output_folder ./output \
  --pellet_scale 2.14774 --hyphae_scale 0.3373 \
  --reference_img path/to/reference.png
```
- `--reference_img` (optional): Used for flat-field correction, it's recommended to use a reference image from the same batch.
- `--pellet_scale`, `--hyphae_scale`: microns/pixel, affects actual unit conversion.

---

## Input and Output

### Pellet Analysis
- Input: Grouped subdirectories under `--input_folder` (e.g., `input/pellet/D353.8-Dox0-24h`, etc.).
- Output: For each group, generated under `--output_folder/<group>/`:
  - Mask images (core/periphery/outline fitting), contour images, fitted ellipse visualizations, etc. as PNG files.
  - Metrics summary `result.csv`, including:
    - `core_average_radius`, `pellet_radius`, `Hyphae_length`
    - `core_projected_area`, `peripheral_projected_area`, `total_projected_area`
    - `fitted_hypothetical_convex_area`
    - `pellet_aspect_ratio`, `pellet_circularity`
    - `core_aspect_ratio`, `core_circularity`
    - `solidity`, `porosity`, `MN`

Note: The current implementation accumulates results from all groups into the same `results` list and writes to each group's `result.csv`. If you need independent statistics for each group, you can reset `results = []` at the beginning of the group loop in `pellet.py`.

### Hyphae/Spore Analysis
- Preprocessing: Gaussian blur + CLAHE enhancement; global thresholding; morphological closing/opening/filling; skeletonization (`skimage.skeletonize`).
- Coloring rules:
  - Green: Tips (endpoints)
  - Red: Segments (2-adjacent)
  - Blue: Branches (≥3-adjacent)
- Statistics: Number of tips, number of branches, total area, total length (in pixels and actual units).
- Spore mode: Filters by length and connected pixels, numbers and statistics for each object.
- Output: For each image, saves original image, binary image, skeleton, colored skeleton, composite image; generates CSV summary during batch processing.

---

## Parameter Description
- `--pellet_scale`: Micron/pixel ratio for pellet analysis (default 2.14774).
- `--hyphae_scale`: Micron/pixel ratio for hyphae analysis (default 0.3373).
- `--reference_img`: Flat-field reference image (optional).

Internally adjustable (in `No_Note/hyphae.py`):
- `global_threshold`, `morph_kernel_size`, `min_object_size`, `pixel_size`, etc.

---

## Methods and Metrics (Brief Description)
- Circularity: `(4πA)/(P^2)` (A = area, P = perimeter).
- Ellipse fitting perimeter: Ramanujan approximation.
- Porosity: Relationship between peripheral area and the ratio of fitted outline minus core area.
- Skeleton length: Accumulated using 8-neighborhood, supports actual unit conversion.

---

## Common Issues
- File writing permissions: If `result.csv` writing fails, the program will attempt to write to the current working directory.
- OpenCV installation: On Windows, you can try `pip install --upgrade pip` before reinstalling.
- Data organization: Ensure the subdirectory structure under `input` matches the description above.

---

## License and Acknowledgments
Unless otherwise stated, this is intended for internal research use only. Please add license information if you plan to publish it publicly.
