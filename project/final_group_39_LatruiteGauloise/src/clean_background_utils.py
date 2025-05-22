import os
import numpy as np
import cv2
from sklearn.cluster import KMeans

from skimage.morphology import remove_small_holes
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from tqdm import tqdm
from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from skimage.morphology import closing, erosion


### -------------- Background Utils -------------- ###


def get_average_hsv(image_path):
    """Compute average HSV color of an image."""
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv_image, axis=(0, 1))
    return avg_hsv


def extract_features_from_folder(folder_path):
    """Extract average HSV from all images in a folder."""
    image_paths = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    features = []
    valid_paths = []
    for path in image_paths:
        try:
            avg_hsv = get_average_hsv(path)
            features.append(avg_hsv)
            valid_paths.append(path)
        except:
            print(f"Error reading {path}")
    return np.array(features), valid_paths


def cluster_images(features, n_clusters=6):
    """Run KMeans clustering on feature array."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels


def show_cluster_examples(labels, image_paths, cluster_id):
    """Show example images from a specific cluster."""
    cluster_indices = np.where(labels == cluster_id)[0]
    # selected = cluster_indices[:max_images]

    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(cluster_indices):
        img = cv2.imread(image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, len(cluster_indices), i + 1)
        plt.imshow(img)
        plt.axis("off")
    plt.suptitle(f"Images from Cluster {cluster_id}")
    plt.show()


def cluster_images_in_folder(folder_path, n_clusters=6):
    """
    Cluster images from a folder based on HSV features.

    Args:
        folder_path (str): Path to folder containing images
        n_clusters (int): Number of clusters to create

    Returns:
        tuple: (clustered_image_lists, labels, centers)
            - clustered_image_lists: List of lists, each containing paths for images in a cluster
            - labels: Cluster assignments for each image
            - centers: Cluster centers from KMeans
    """
    # Extract features
    features, image_paths = extract_features_from_folder(folder_path)

    # Cluster the images
    labels = cluster_images(features, n_clusters=n_clusters)

    # Initialize a list of lists to hold image paths per cluster
    clustered_image_lists = [[] for _ in range(n_clusters)]

    # Assign each image path to the appropriate cluster list
    for path, label in zip(image_paths, labels):
        clustered_image_lists[label].append(path)

    return clustered_image_lists


def plot_cluster(clustered_image_lists, cluster_index, figsize=(15, None), cols=4):
    """
    Plot all images from a specific cluster.

    Args:
        clustered_image_lists (list): List of lists containing image paths for each cluster
        cluster_index (int): Index of the cluster to plot
        figsize (tuple): Figure size (width, height). If height is None, it will be calculated based on rows
        cols (int): Number of columns in the grid layout
    """
    # Get the list of images for the chosen cluster
    cluster_images = clustered_image_lists[cluster_index]
    num_images = len(cluster_images)

    # Calculate rows needed for the layout
    rows = (num_images + cols - 1) // cols

    # Set the figure height if not specified
    if figsize[1] is None:
        figsize = (figsize[0], 3 * rows)

    # Create the figure with the specified size
    plt.figure(figsize=figsize)

    # Plot each image in the cluster
    for i, path in enumerate(cluster_images):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.axis("off")

    # Add a title and adjust layout
    plt.suptitle(f"Images from Cluster {cluster_index}")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


### -------------- Background Utils -------------- ###


def process_image(
    img_path: str,
    downsample_factor: int = 2,
    bilateral_d: int = 9,
    bilateral_sigmaColor: float = 5,
    bilateral_sigmaSpace: float = 5,
    adapt_block: int = 55,
    adapt_C: int = 2,
    canny_t1: int = 10,
    canny_t2: int = 30,
    hole_area_threshold: int = 10000,
    min_area: int = 8000,
    max_area: int = 20000,
):
    # 1) Load
    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Could not load {img_path}")
        return None, None  # Return None if image loading fails

    img_resized = cv2.resize(
        img,
        None,
        fx=1 / downsample_factor,
        fy=1 / downsample_factor,
        interpolation=cv2.INTER_AREA,
    )

    # 3) Grayscale & Bilateral Blur
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(
        gray, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace
    )

    # 4) Thresholds + Edges
    _, th_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    th_adapt = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        adapt_block,
        adapt_C,
    )
    edges = cv2.Canny(blur, canny_t1, canny_t2)

    # 5) Combine masks
    mask = cv2.bitwise_or(th_otsu, th_adapt)
    mask = cv2.bitwise_or(mask, edges)

    # 6) Pad & flood-fill background from (0,0)
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    h2, w2 = padded.shape
    flood = padded.copy()
    bgmask = np.zeros((h2 + 2, w2 + 2), np.uint8)
    cv2.floodFill(flood, bgmask, (0, 0), 255)
    flood_inv = cv2.bitwise_not(flood)
    filled_pad = cv2.bitwise_or(padded, flood_inv)
    mask_filled = filled_pad[1:-1, 1:-1]

    # 7) Remove small holes via skimage
    mask_bool = mask_filled.astype(bool)
    mask_no_holes_b = remove_small_holes(
        mask_bool, area_threshold=hole_area_threshold, connectivity=2
    )
    mask_no_holes = mask_no_holes_b.astype(np.uint8) * 255

    # 8) Morphological close & open to clean edges
    kern_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kern_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask_close = cv2.morphologyEx(
        mask_no_holes, cv2.MORPH_CLOSE, kern_close, iterations=1
    )
    mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kern_open, iterations=1)

    # 9) Area filtering
    cnts, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_final = np.zeros_like(mask_open)
    for c in cnts:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            cv2.drawContours(mask_final, [c], -1, 255, cv2.FILLED)

    # 10) Apply mask to original resized image
    segmented = cv2.bitwise_and(img_resized, img_resized, mask=mask_final)

    # 11) Save outputs
    # cv2.imwrite(out_image_path, segmented)
    # if out_mask_path:
    #   cv2.imwrite(out_mask_path, mask_final)

    # 12) Report
    final_cnts_after_filter, _ = cv2.findContours(
        mask_final, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    print(
        f"[OK] {os.path.basename(img_path)} → {len(final_cnts_after_filter)} objects passing area filter"
    )

    return img_resized, mask_final  # Return for feature extraction


def compute_mask(img_gray: np.ndarray, name: str) -> np.ndarray:

    edges = canny(img_gray)

    if "White" in name or "Comtesse" in name:
        result = hough_ellipse(edges, accuracy=20, threshold=4, min_size=4)
        result.sort(order="accumulator")

        best = list(result[-1])
        yc, xc, a, b = (int(round(x)) for x in best[1:5])
        orientation = best[5]

        cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
        edges = np.zeros(img_gray.shape)
        edges[cy, cx] = 1

    kernel = np.ones((7, 7))
    edges = closing(edges, kernel)
    mask = remove_small_holes(edges.astype(dtype=np.bool), area_threshold=500000)
    mask = erosion(mask, kernel)

    return mask


def compute_masked_histogram(image, mask, bins=8, color_space="HSV"):
    """
    Compute a normalized 3D histogram of `image` restricted to `mask`.
    Both image and mask should already be at the desired (potentially downscaled) resolution.
    """
    if image is None or image.size == 0 or mask is None or mask.size == 0:
        # Return a zero histogram or handle error appropriately
        print(f"[WARN] Cannot compute histogram for empty image or mask.")
        return np.zeros((bins, bins, bins), dtype=np.float32)

    # convert color space
    if color_space == "HSV":
        img_conv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_space == "LAB":
        img_conv = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    elif color_space == "BGR":
        img_conv = image
    else:
        raise ValueError(f"Unsupported color space: {color_space}")

    # ensure mask is single‐channel 8‐bit
    mask8 = (mask > 0).astype(np.uint8)

    # Check if mask has any non-zero pixels
    if cv2.countNonZero(mask8) == 0:
        # print(f"[INFO] Mask is empty. Returning zero histogram.")
        return np.zeros((bins, bins, bins), dtype=np.float32)

    hist = cv2.calcHist([img_conv], [0, 1, 2], mask8, [bins] * 3, [0, 256] * 3)
    return cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)


def process_references(
    ref_dir,
    exts=[".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"],
    hist_bins=25,
    hist_color_space="HSV",
    downsample_factor=2,
):
    """
    Process reference images and compute their histograms.

    Args:
        ref_dir (str): Directory containing reference images
        exts (list): List of valid file extensions
        hist_bins (int): Number of histogram bins
        hist_color_space (str): Color space for histogram computation
        downsample_factor (int): Factor to downsample images

    Returns:
        tuple: (reference_histograms, reference_ids, reference_labels)
            - reference_histograms: List of histograms for each reference
            - reference_ids: List of IDs for each reference
            - reference_labels: List of labels (filenames without extension)
    """
    # Initialize empty lists
    reference_histograms = []
    reference_ids = []
    reference_labels = []

    # Check if reference directory exists
    if not os.path.exists(ref_dir):
        print(f"‼️ Reference directory not found: {ref_dir}")
        return reference_histograms, reference_ids, reference_labels

    # Get reference files
    ref_files = sorted(
        [f for f in os.listdir(ref_dir) if os.path.splitext(f)[1].lower() in exts]
    )

    # Process each reference image
    for idx, name in enumerate(tqdm(ref_files, desc="Processing References"), start=1):
        path = os.path.join(ref_dir, name)
        ref_img_orig = cv2.imread(path)

        if ref_img_orig is None:
            print(f"‼️ Could not load reference {path}")
            continue

        # 1. Downsample original reference image
        h_ref_orig, w_ref_orig = ref_img_orig.shape[:2]
        new_w_ref = max(1, w_ref_orig // downsample_factor)
        new_h_ref = max(1, h_ref_orig // downsample_factor)
        ref_img_ds = cv2.resize(
            ref_img_orig, (new_w_ref, new_h_ref), interpolation=cv2.INTER_AREA
        )

        # 2. Generate mask from downsampled grayscale reference image
        gray_ref_ds = cv2.cvtColor(ref_img_ds, cv2.COLOR_BGR2GRAY)
        ref_mask_ds = compute_mask(gray_ref_ds, name)

        # Check if mask is empty
        if not np.any(ref_mask_ds):
            print(
                f"  [WARN] Empty mask generated for reference {name} (on downscaled image), skipping."
            )
            continue

        # 3. Compute masked histogram on downsampled versions
        h = compute_masked_histogram(
            ref_img_ds, ref_mask_ds, bins=hist_bins, color_space=hist_color_space
        )

        # Store results
        reference_histograms.append(h)
        reference_ids.append(idx)
        reference_labels.append(os.path.splitext(name)[0])

    print(
        f"Loaded {len(reference_histograms)} reference histograms from {len(ref_files)} files."
    )
    return reference_histograms, reference_ids, reference_labels


def process_clustered_images_and_extract_features(
    clustered_image_lists,
    reference_histograms,
    downsample_factor=4,
    process_image_min_area=7000,
    process_image_max_area=20000,
    # Add other process_image params if needed to be configurable
    hist_bins=25,
    hist_color_space="HSV",
    num_top_hist_distances=5,
):
    """
    Processes images grouped by clusters, segments objects, and extracts features.

    Args:
        clustered_image_lists (list of lists): Image paths grouped by cluster.
        reference_histograms (list): List of reference histograms.
        reference_ids (list): List of reference IDs.
        reference_labels (list): List of reference labels.
        input_image_base_folder (str): The root directory from which image paths in
                                       clustered_image_lists are relative. Or where they are.
        output_folder (str): Path to save segmented images.
        mask_folder (str, optional): Path to save masks. Defaults to None.
        blobs_for_labeling_dir (str, optional): Dir to save blob images for GUI.
        downsample_factor (int, optional): Downsample factor for process_image.
        process_image_min_area (int, optional): Min area for objects in process_image.
        process_image_max_area (int, optional): Max area for objects in process_image.
        hist_bins (int, optional): Number of bins for blob histograms.
        hist_color_space (str, optional): Color space for blob histograms.
        num_top_hist_distances (int, optional): Number of top histogram distances to store.

    Returns:
        list: A list of dictionaries, where each dictionary contains features for a detected blob.
    """
    all_blob_features = []

    # Flatten the list of image paths for a single progress bar,
    # or iterate cluster by cluster if specific per-cluster actions are needed.
    all_image_paths_to_process = [
        img_path for cluster_list in clustered_image_lists for img_path in cluster_list
    ]

    print(
        f"\nProcessing {len(all_image_paths_to_process)} images from clustered lists..."
    )
    for img_path_relative in tqdm(
        all_image_paths_to_process, desc="Processing Clustered Images"
    ):

        inp = img_path_relative

        fn = os.path.basename(inp)  # Get filename for output paths

        # Process the image to get segmented objects and the mask
        # process_image is expected to be defined elsewhere (e.g., in background_utils.py)
        img_at_target_height, mask_at_target_height = process_image(
            inp,
            downsample_factor=downsample_factor,
            min_area=process_image_min_area,
            max_area=process_image_max_area,
        )

        if img_at_target_height is None or mask_at_target_height is None:
            print(f"Skipping feature extraction for {fn} due to processing error.")
            continue

        if cv2.countNonZero(mask_at_target_height) == 0:
            # print(f"[INFO] No blobs found in {fn} by process_image (empty final mask).")
            continue

        # Label connected components (blobs) in the mask
        lbl_full_mask = label(mask_at_target_height > 0)
        regions = regionprops(lbl_full_mask)

        if not regions:
            continue

        for region in regions:
            area = region.area
            # Ensure minor_axis_length is not zero before division
            elongation = (
                region.major_axis_length / region.minor_axis_length
                if region.minor_axis_length > 0
                else 0
            )
            solidity = region.solidity
            extent = region.extent
            perimeter = region.perimeter
            circularity = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0

            current_blob_mask_at_target_height = (lbl_full_mask == region.label).astype(
                np.uint8
            ) * 255

            # Hu Moments
            moments = cv2.moments(current_blob_mask_at_target_height, binaryImage=True)
            hu_moments_transformed = np.zeros(7, dtype=np.float32)
            if moments["m00"] != 0:
                hu_moments_raw = cv2.HuMoments(moments)
                for i in range(0, 7):
                    if hu_moments_raw[i, 0] != 0:
                        hu_moments_transformed[i] = (
                            -1
                            * np.copysign(1.0, hu_moments_raw[i, 0])
                            * np.log10(abs(hu_moments_raw[i, 0]))
                        )
                    else:
                        hu_moments_transformed[i] = 0
            hu_moments_list = hu_moments_transformed.flatten().tolist()

            # Blob Histogram
            # compute_masked_histogram is expected to be defined elsewhere
            blob_hist = compute_masked_histogram(
                img_at_target_height,
                current_blob_mask_at_target_height,
                bins=hist_bins,
                color_space=hist_color_space,
            )

            # Histogram Distances
            top_n_hist_distances = [float("inf")] * num_top_hist_distances
            if reference_histograms:
                all_histogram_distances_to_refs = [
                    cv2.compareHist(blob_hist, h_ref, cv2.HISTCMP_BHATTACHARYYA)
                    for h_ref in reference_histograms
                ]
                if all_histogram_distances_to_refs:
                    sorted_distances_with_indices = sorted(
                        enumerate(all_histogram_distances_to_refs), key=lambda x: x[1]
                    )
                    for i in range(
                        min(num_top_hist_distances, len(sorted_distances_with_indices))
                    ):
                        _, dist_val = sorted_distances_with_indices[i]
                        top_n_hist_distances[i] = dist_val

            # Populate Feature Dictionary
            feature_dict = {
                "image_filename": fn,
                "blob_label_in_image": region.label,
                "area": area,
                "elongation": elongation,
                "solidity": solidity,
                "extent": extent,
                "perimeter": perimeter,
                "circularity": circularity,
                "hu_moments": hu_moments_list,
            }
            for i in range(num_top_hist_distances):
                feature_dict[f"hist_dist_top_{i+1}"] = top_n_hist_distances[i]

            all_blob_features.append(feature_dict)

    return all_blob_features
