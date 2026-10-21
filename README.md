# Satellite Change AI 🛰️

A computer vision project exploring how satellite imagery from different time periods can be compared to detect changes in the Earth's surface.

## Project Goals

- Visualize satellite images captured at different times.
- Compare earlier and later images of the same location.
- Explore traditional image-difference techniques.
- Progress toward machine-learning and deep-learning-based change detection.
- Build an interactive application for visualizing results.

## Dataset

This project uses the LEVIR-CD building change detection dataset.

The dataset provides paired remote-sensing images and corresponding ground-truth change masks.

Dataset website: https://justchenhao.github.io/LEVIR/

Raw dataset files are excluded from this repository because of their size.

## Tech Stack

- Python
- OpenCV
- NumPy
- Matplotlib
- Git and GitHub

## Current Progress

- [x] Downloaded and extracted the dataset
- [x] Verified a matching image pair and ground-truth mask
- [x] Visualized the before, after, and label images
- [ ] Implement a pixel-difference baseline
- [ ] Evaluate predictions against ground truth
- [ ] Explore machine learning and deep learning
- [ ] Build a web application

## Run Locally

1. Install Python and Git.
2. Install the required libraries:

   `pip install numpy matplotlib opencv-python`

3. Download the LEVIR-CD dataset and extract the relevant `A`, `B`, and `label` folders into the project root.
4. Run:

   `python main.py`

## Status

Early development — currently focused on dataset exploration and baseline change detection.