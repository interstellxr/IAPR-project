import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from typing import Sequence, Optional, List, Tuple, Union

from sklearn.cluster import DBSCAN
from skimage.draw import ellipse_perimeter
from skimage.transform import hough_ellipse
from skimage.morphology import closing, opening, remove_small_holes, erosion
from skimage.feature import (
    blob_log,
    local_binary_pattern,
    hog,
    canny,
    graycomatrix,
    graycoprops,
)


def load_images(dir: str, downsampling_factor: int) -> Union[List[str], np.ndarray]:
    # Get sorted list of JPG files
    names = sorted([n for n in os.listdir(dir) if n.endswith(".JPG")])

    # Pre-allocate numpy array for images
    first_img = cv2.imread(os.path.join(dir, names[0]), cv2.IMREAD_COLOR_RGB)
    y, x, _ = first_img.shape
    new_shape = (y // downsampling_factor, x // downsampling_factor, 3)
    downsampled_images = np.empty((len(names), *new_shape), dtype=np.uint8)

    # Load and resize images in one pass
    for i, name in enumerate(names):
        img = cv2.imread(os.path.join(dir, name), cv2.IMREAD_COLOR_RGB)
        downsampled_images[i] = cv2.resize(
            img, (x // downsampling_factor, y // downsampling_factor)
        )

    return names, downsampled_images


def compute_mask(img_gray: np.ndarray, name: str) -> np.ndarray:
    edges = canny(img_gray, sigma=1)

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
    mask = remove_small_holes(edges.astype(dtype=np.bool), area_threshold=500_000)
    mask = erosion(mask, kernel)

    return mask


def compute_ref_histogram(
    images: Sequence[np.ndarray], masks: Sequence[np.ndarray]
) -> List[np.ndarray]:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    hist_list = []

    for img, mask in zip(images, masks):
        mask_uint8 = (mask.astype(np.uint8)) * 255
        eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)
        hist = cv2.calcHist([img], [0, 1, 2], eroded_mask, [16, 16, 16], [0, 256] * 3)
        hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten()
        hist_list.append(hist)
    return hist_list


def compute_histogram(
    image: np.ndarray,
    mask: Optional[np.ndarray] = None,
    kernel_size: int = 11,
    hist_params: dict = dict(
        channels=[0, 1, 2],
        histSize=[16, 16, 16],
        ranges=[0, 256, 0, 256, 0, 256],
    ),
) -> np.ndarray:

    hist_args = hist_params
    kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        if kernel_size > 0
        else None
    )

    m = None
    if mask is not None:
        m = (mask.astype(np.uint8)) * 255
        if kernel is not None:
            m = cv2.erode(m, kernel, iterations=1)

    hist = cv2.calcHist([image], mask=m, **hist_args)
    hist = cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten().astype(np.float32)
    return hist


def compute_histograms(
    images: Sequence[np.ndarray],
    masks: Optional[Sequence[np.ndarray]] = None,
    *,
    select_channels: List[int] = [0, 1, 2],
    bins: int = 16,
    ker_size: int = 11,
) -> List[np.ndarray]:
    if masks is not None and len(images) != len(masks):
        raise ValueError("`images` and `masks` must have the same length.")

    hist_args = dict(
        channels=select_channels,
        histSize=[12] * len(select_channels),
        ranges=[0, 256] * len(select_channels),
    )
    ell_kernel = (
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ker_size, ker_size))
        if ker_size > 0
        else None
    )

    hist_list: list[np.ndarray] = []

    for idx, img in enumerate(images):
        m = None
        if masks is not None:
            m = (masks[idx].astype(np.uint8)) * 255
            if ell_kernel is not None:
                m = cv2.erode(m, ell_kernel, iterations=1)

        # img = cv2.GaussianBlur(img, (5, 5), 5)

        hist = cv2.calcHist([img], mask=m, **hist_args)
        hist = (
            cv2.normalize(hist, hist, norm_type=cv2.NORM_L1)
            .flatten()
            .astype(np.float32)
        )
        hist_list.append(hist)
    return hist_list


import cv2
import numpy as np
from typing import Sequence, List, Tuple, Dict


# ────────────────────────────────────────────────────────────────────────────
#  1.  Re-usable histogram helper (L1-normalised, optional mask)
# ────────────────────────────────────────────────────────────────────────────
def _hist_3d(
    img: np.ndarray,
    bins: int = 16,
    mask: np.ndarray = None,
) -> np.ndarray:
    hist = cv2.calcHist(
        [img],
        [0, 1, 2],
        mask,
        [bins, bins, bins],
        [0, 256] * 3,
    )
    return cv2.normalize(hist, hist, norm_type=cv2.NORM_L1).flatten().astype(np.float32)


# ────────────────────────────────────────────────────────────────────────────
#  2.  Build reference histograms (RGB, HSV, Lab) *inside* masks
# ────────────────────────────────────────────────────────────────────────────
def prepare_reference_histograms(
    ref_images: Sequence[np.ndarray],
    ref_masks: Sequence[np.ndarray],
    *,
    bins: int = 16,
    erode_kernel: int = 11,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Returns three parallel lists of histograms (RGB/HSV/Lab) computed
    only on the eroded foreground masks.
    """
    if len(ref_images) != len(ref_masks):
        raise ValueError("ref_images and ref_masks must have same length")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_kernel, erode_kernel))
    hists_rgb, hists_hsv, hists_lab = [], [], []

    for img, m in zip(ref_images, ref_masks):
        m_uint8 = (m.astype(np.uint8)) * 255
        m_uint8 = cv2.erode(m_uint8, kernel, iterations=1)

        hists_rgb.append(_hist_3d(img, bins=bins, mask=m_uint8))
        hists_hsv.append(
            _hist_3d(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), bins=bins, mask=m_uint8)
        )
        hists_lab.append(
            _hist_3d(cv2.cvtColor(img, cv2.COLOR_RGB2LAB), bins=bins, mask=m_uint8)
        )
    return hists_rgb, hists_hsv, hists_lab


# ────────────────────────────────────────────────────────────────────────────
#  3.  Patch-side colour-space histograms (no mask)
# ────────────────────────────────────────────────────────────────────────────
def patch_histograms(patch: np.ndarray, bins: int = 16) -> Dict[str, np.ndarray]:
    return {
        "rgb": _hist_3d(patch, bins=bins),
        "hsv": _hist_3d(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV), bins=bins),
        "lab": _hist_3d(cv2.cvtColor(patch, cv2.COLOR_BGR2LAB), bins=bins),
    }


# ────────────────────────────────────────────────────────────────────────────
#  4.  Sliding-window compare: best similarity across RGB / HSV / Lab
# ────────────────────────────────────────────────────────────────────────────
def sliding_window_compare_multispace(
    image: np.ndarray,
    ref_hists_rgb: Sequence[np.ndarray],
    ref_hists_hsv: Sequence[np.ndarray],
    ref_hists_lab: Sequence[np.ndarray],
    *,
    window_size: int = 64,
    stride: int = 16,
    bins: int = 16,
) -> np.ndarray:
    H, W, _ = image.shape
    heatmap = np.zeros((H // stride, W // stride), dtype=np.float32)

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            patch = image[y : y + window_size, x : x + window_size]
            h = patch_histograms(patch, bins=bins)

            sim_rgb = 1.0 - min(
                cv2.compareHist(h["rgb"], r, cv2.HISTCMP_BHATTACHARYYA)
                for r in ref_hists_rgb
            )
            sim_hsv = 1.0 - min(
                cv2.compareHist(h["hsv"], r, cv2.HISTCMP_BHATTACHARYYA)
                for r in ref_hists_hsv
            )
            sim_lab = 1.0 - min(
                cv2.compareHist(h["lab"], r, cv2.HISTCMP_BHATTACHARYYA)
                for r in ref_hists_lab
            )

            heatmap[y // stride, x // stride] = max(sim_rgb, sim_hsv, sim_lab)

    return heatmap


def sliding_window_compare(
    image,
    reference_histograms,
    window_size=64,
    stride=16,
    hist_params: dict = dict(
        channels=[0, 1, 2],
        histSize=[16, 16, 16],
        ranges=[0, 256, 0, 256, 0, 256],
    ),
):
    H, W, _ = image.shape
    heatmap = np.zeros((H // stride, W // stride))

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            patch = image[y : y + window_size, x : x + window_size]

            patch_hist = compute_histogram(
                patch, mask=None, kernel_size=11, hist_params=hist_params
            )

            # Compare to all reference histograms
            min_dist = min(
                [
                    cv2.compareHist(patch_hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)
                    for ref_hist in reference_histograms
                ]
            )
            similarity = 1.0 - min_dist  # invert for similarity

            heatmap[y // stride, x // stride] = similarity

    return heatmap


def compute_blobs(
    heatmap: np.ndarray,
    min_sigma: float = 2.0,
    max_sigma: float = 15.0,
    thr: float = 0.10,
    avg_R: float = 32.0,
    merge_policy: str = "mean",  # "mean" | "max" | "fixed"
) -> list[tuple[float, float, float]]:
    """
    Return a list of (y, x, R) circles around chocolate hot-spots in `heatmap`.

    • min_sigma, max_sigma, thr → parameters for skimage.feature.blob_log.
    • avg_R   : approximate chocolate radius in *heat-map* pixels.
    • merge_policy :
        - "mean" : R = average of member radii
        - "max"  : R = largest member radius
        - "fixed": R = avg_R
    """
    # -------- 1)  find raw LoG blobs ----------------
    raw = blob_log(heatmap, min_sigma=min_sigma, max_sigma=max_sigma, threshold=thr)
    if raw.size == 0:
        return []

    # skimage returns σ in the 3rd column → convert to radius
    raw[:, 2] = np.sqrt(2) * raw[:, 2]

    # -------- 2)  cluster blobs that lie closer than avg_R ----------
    #   eps = avg_R  ⇒ centres < avg_R pixels apart are merged
    db = DBSCAN(eps=avg_R * 1.2, min_samples=1).fit(raw[:, :2])
    labels = db.labels_

    merged = []
    for L in np.unique(labels):
        member = raw[labels == L]  # (n_i, 3)

        y_c = member[:, 0].mean()
        x_c = member[:, 1].mean()

        if merge_policy == "mean":
            R_c = member[:, 2].mean()
        elif merge_policy == "max":
            R_c = member[:, 2].max()
        else:  # fixed
            R_c = avg_R

        merged.append((y_c, x_c, R_c))

    return merged


def extract_crops(
    images: np.ndarray, blobs: list, crop_size: int, window_size: int, stride: int
) -> list[list[np.ndarray]]:
    all_crops = []

    for idx, img in enumerate(images):
        img_crops = []
        for y, x, r in blobs[idx]:
            y, x = int(y), int(x)

            y, x = y * stride + window_size // 2, x * stride + window_size // 2

            # Calculate patch boundaries
            y_start = max(0, y - crop_size // 2)
            y_end = min(img.shape[0], y + crop_size // 2)
            x_start = max(0, x - crop_size // 2)
            x_end = min(img.shape[1], x + crop_size // 2)

            # Extract patch
            crop = img[y_start:y_end, x_start:x_end]

            # Pad if necessary to maintain square shape
            if crop.shape[0] != crop_size or crop.shape[1] != crop_size:
                padded_crop = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
                h, w = crop.shape[:2]
                padded_crop[:h, :w] = crop
                crop = padded_crop

            img_crops.append(crop)
        all_crops.append(img_crops)

    return all_crops


def plot_heatmap_result(
    filtered_heatmaps, images_rgb, blobs, window_size, stride, idx: int
):
    resized_heatmap = cv2.resize(
        filtered_heatmaps[idx],
        (images_rgb[idx].shape[1], images_rgb[idx].shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    # Create a translation matrix to shift by half stride
    M = np.float32([[1, 0, window_size // 2], [0, 1, window_size // 2]])
    resized_heatmap = cv2.warpAffine(
        resized_heatmap, M, (resized_heatmap.shape[1], resized_heatmap.shape[0])
    )

    plt.figure(figsize=(15, 9))

    # Plot original image with heatmap overlay
    plt.imshow(images_rgb[idx])
    plt.imshow(
        resized_heatmap, cmap="hot", alpha=0.5
    )  # Overlay heatmap with 50% transparency

    # Add circles for detected blobs
    for y, x, r in blobs[idx]:
        # Scale coordinates and radius by stride
        y_scaled, x_scaled = (
            y * stride + window_size // 2,
            x * stride + window_size // 2,
        )
        r_scaled = r * stride
        circle = plt.Circle(
            (x_scaled, y_scaled), r_scaled, color="cyan", fill=False, linewidth=2
        )
        plt.gca().add_patch(circle)

    plt.title("Original Image with Heatmap Overlay")
    plt.colorbar(label="Similarity")
    plt.tight_layout()
    plt.axis("off")
    plt.show()


# --------------------------------------------------------------------------
# Helper: Gabor energy on one channel
# --------------------------------------------------------------------------
def _gabor_energy(
    channel: np.ndarray,
    freqs=(0.2, 0.4),
    thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> list:
    """
    Apply a small Gabor filter bank to a single channel and return
    the mean magnitude (= energy) for each filter.
    """
    energies = []
    for f in freqs:
        for t in thetas:
            ksize = 17  # reasonable on 64×64
            sigma = 3.0
            lambd = 1 / f
            kernel = cv2.getGaborKernel(
                (ksize, ksize), sigma, t, lambd, gamma=0.5, psi=0, ktype=cv2.CV_32F
            )
            resp = cv2.filter2D(channel.astype(np.float32), cv2.CV_32F, kernel)
            energies.append(float(np.mean(np.abs(resp))))
    return energies  # length = len(freqs) * len(thetas)


# --------------------------------------------------------------------------
# Helper: single-patch descriptor
# --------------------------------------------------------------------------
def _single_crop_features(
    patch_rgb: np.ndarray,
    *,
    hist_bins: int = 32,
    lbp_P: int = 8,
    lbp_R: int = 1,
    hog_pixels_cell: int = 8,
    gabor_freqs=(0.2, 0.4),
    gabor_thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> np.ndarray:

    feat = []

    # --- 1. colour statistics --------------------------------------------
    # Lab space
    lab = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2LAB)
    feat.extend(lab.reshape(-1, 3).mean(0))  # mean Lab (3)
    feat.extend(lab.reshape(-1, 3).std(0))  # std  Lab (3)

    # RGB space (mean + std)
    feat.extend(patch_rgb.reshape(-1, 3).mean(0))  # mean RGB (3)
    feat.extend(patch_rgb.reshape(-1, 3).std(0))  # std  RGB (3)

    # Lab histograms (3×32)
    for ch in range(3):
        h = cv2.calcHist([lab], [ch], None, [hist_bins], [0, 256])
        h = cv2.normalize(h, h).flatten()
        feat.extend(h)  # 96

    # RGB histograms (3×32)
    for ch in range(3):
        h = cv2.calcHist([patch_rgb], [ch], None, [hist_bins], [0, 256])
        h = cv2.normalize(h, h).flatten()
        feat.extend(h)  # +96

    # --- 2. texture -------------------------------------------------------
    gray = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2GRAY)

    # LBP histogram (10)
    lbp = local_binary_pattern(gray, lbp_P, lbp_R, method="uniform")
    lbp_hist, _ = np.histogram(
        lbp, bins=np.arange(0, lbp_P + 3), range=(0, lbp_P + 2), density=True
    )
    feat.extend(lbp_hist)  # 10

    # HOG (4×4 cells × 9 bins = 144)
    hog_vec = hog(
        gray,
        orientations=9,
        pixels_per_cell=(hog_pixels_cell, hog_pixels_cell),
        cells_per_block=(1, 1),
        feature_vector=True,
    )
    feat.extend(hog_vec)

    # Haralick GLCM features (4)
    gray8 = (gray // 32).astype(np.uint8)
    glcm = graycomatrix(
        gray8, distances=[1], angles=[0], levels=8, symmetric=True, normed=True
    )
    for prop in ("contrast", "dissimilarity", "homogeneity", "energy"):
        feat.append(float(graycoprops(glcm, prop)[0, 0]))

    # --- 3. Gabor energies per RGB channel --------------------------------
    for c in range(3):
        energies = _gabor_energy(
            patch_rgb[:, :, c], freqs=gabor_freqs, thetas=gabor_thetas
        )  # 8 energies / channel
        feat.extend(energies)  # 24 total

    return np.asarray(feat, dtype=np.float32)


def extract_crop_features(
    patches: list[np.ndarray],
    *,
    hist_bins: int = 32,
    lbp_P: int = 8,
    lbp_R: int = 1,
    hog_pixels_cell: int = 8,
    gabor_freqs=(0.2, 0.4),
    gabor_thetas=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
) -> np.ndarray:
    """
    Compute a rich colour+texture feature matrix for a list of 64×64 RGB patches.

    Returns
    -------
    feats : ndarray shape (N, F)
        Where F =  6(Lab mean/std) + 6(RGB mean/std)
                  + 96(Lab hist)   + 96(RGB hist)
                  + 10(LBP)        + 144(HOG)
                  + 4(GLCM)        + 24(Gabor energies)  = **386**
    """
    feats = [
        _single_crop_features(
            p,
            hist_bins=hist_bins,
            lbp_P=lbp_P,
            lbp_R=lbp_R,
            hog_pixels_cell=hog_pixels_cell,
            gabor_freqs=gabor_freqs,
            gabor_thetas=gabor_thetas,
        )
        for p in patches
    ]
    return np.vstack(feats)


def extract_reference_crops(
    img_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    box_size: int = 64,
    stride: int = 8,
    min_coverage: float = 0.75,
    min_patches: int = 3,
    max_patches: int = 7,
) -> list[tuple[int, int, int, int]]:
    """
    Find 64×64 windows that cover at least `min_coverage` of chocolate pixels.

    Parameters
    ----------
    img_rgb     : H×W×3 uint8  – reference image (unused here but kept for API).
    mask        : H×W bool/uint8 – binary mask of the chocolate in `img_rgb`.
    box_size    : side length of the square window (default 64).
    stride      : sliding offset in pixels (default 8).
    min_coverage: required fraction of FG pixels inside a window (≥ 0.75).
    min_patches : guarantee at least this many boxes (relaxes threshold if needed).
    max_patches : return at most this many boxes (top coverage first).

    Returns
    -------
    boxes : list of (y1, x1, y2, x2) tuples – coordinates in image pixels.
    """

    mask_bool = mask.astype(bool)
    H, W = mask_bool.shape

    # Integral image of mask to compute FG pixel counts in O(1)
    integ = cv2.integral(mask_bool.astype(np.uint8))  # shape (H+1, W+1)

    def fg_pixels(y, x):
        y2, x2 = y + box_size, x + box_size
        return integ[y2, x2] - integ[y, x2] - integ[y2, x] + integ[y, x]

    # Scan grid
    boxes = []
    for y in range(0, H - box_size + 1, stride):
        for x in range(0, W - box_size + 1, stride):
            fg = fg_pixels(y, x)
            if fg / (box_size * box_size) >= min_coverage:
                boxes.append((y, x, y + box_size, x + box_size, fg))

    # If too few, relax coverage threshold progressively
    cov = min_coverage
    while len(boxes) < min_patches and cov > 0.40:  # don’t go too low
        cov -= 0.05
        boxes = []
        for y in range(0, H - box_size + 1, stride):
            for x in range(0, W - box_size + 1, stride):
                fg = fg_pixels(y, x)
                if fg / (box_size * box_size) >= cov:
                    boxes.append((y, x, y + box_size, x + box_size, fg))

    if not boxes:  # fallback: centre crop
        cy, cx = np.column_stack(np.nonzero(mask_bool)).mean(0).astype(int)
        y1 = max(0, cy - box_size // 2)
        x1 = max(0, cx - box_size // 2)
        y1 = min(y1, H - box_size)
        x1 = min(x1, W - box_size)
        boxes = [(y1, x1, y1 + box_size, x1 + box_size, box_size**2)]

    # sort by descending foreground count and drop extra field
    boxes = sorted(boxes, key=lambda b: b[4], reverse=True)[:max_patches]
    return [(y1, x1, y2, x2) for y1, x1, y2, x2, _ in boxes]
