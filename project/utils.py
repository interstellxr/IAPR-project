from skimage.color import rgb2hsv
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects, binary_dilation
from skimage.transform import rotate, resize
from sklearn.metrics.pairwise import euclidean_distances
from skimage.measure import regionprops, label
from skimage.measure import find_contours
from skimage.util import img_as_bool

from skimage.feature import (
    local_binary_pattern, graycomatrix, graycoprops, hog
)
from scipy.stats import skew
import itertools

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from typing import Callable
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

def linear_interpolation(contours: np.ndarray, n_samples: int = 80):
    N = len(contours)
    contours_inter = np.zeros((N, n_samples, 2))
    
    for i in range(N):
        x = contours[i][:, 0]
        y = contours[i][:, 1]

        t = np.zeros(len(x))
        t[1:] = np.cumsum(np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2))

        t_prime = np.linspace(0, t[-1], n_samples)

        contours_inter[i, :, 0] = np.interp(t_prime, t, x)
        contours_inter[i, :, 1] = np.interp(t_prime, t, y)
    
    return contours_inter
def sliding_window_compare(image_rgb, reference_histograms, window_size=64, stride=16, method='hist'):
    H, W, _ = image_rgb.shape
    heatmap = np.zeros((H // stride, W // stride))

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            patch = image_rgb[y:y+window_size, x:x+window_size]

            # Compute feature of patch
            if method == 'hist':
                patch_hist = cv2.calcHist([patch], [0, 1, 2], None, [16, 16, 16], [0, 256]*3)
                patch_hist = cv2.normalize(patch_hist, patch_hist).flatten().astype(np.float32)

                # Compare to all reference histograms
                min_dist = min([
                    cv2.compareHist(patch_hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)
                    for ref_hist in reference_histograms
                ])
                similarity = 1.0 - min_dist  # invert for similarity
                
            heatmap[y // stride, x // stride] = similarity

    return heatmap
def sliding_window_classify_and_count(image_rgb, reference_histograms, reference_names, 
                                      window_size=64, stride=16, method='hist',
                                      similarity_threshold=0.7, cluster=False):
    H, W, _ = image_rgb.shape
    heatmap = np.zeros((H // stride, W // stride))
    count_dict = defaultdict(int)

    positions_by_type = defaultdict(list)  

    for y in range(0, H - window_size + 1, stride):
        for x in range(0, W - window_size + 1, stride):
            patch = image_rgb[y:y+window_size, x:x+window_size]

            if method == 'hist':
                patch_hist = cv2.calcHist([patch], [0, 1, 2], None, [16, 16, 16], [0, 256]*3)
                patch_hist = cv2.normalize(patch_hist, patch_hist).flatten().astype(np.float32)

                similarities = [1.0 - cv2.compareHist(patch_hist, ref_hist, cv2.HISTCMP_BHATTACHARYYA)
                                for ref_hist in reference_histograms]
                
                best_idx = int(np.argmax(similarities))
                best_sim = similarities[best_idx]

                if best_sim > similarity_threshold:
                    count_dict[reference_names[best_idx]] += 1
                    positions_by_type[reference_names[best_idx]].append((x, y))

                heatmap[y // stride, x // stride] = best_sim

    if cluster:
        clustered_counts = {}
        for choco_type, positions in positions_by_type.items():
            if not positions:
                clustered_counts[choco_type] = 0
                continue
            X = np.array(positions)
            clustering = DBSCAN(eps=window_size, min_samples=1).fit(X)
            clustered_counts[choco_type] = len(set(clustering.labels_))
        return heatmap, clustered_counts

    return heatmap, dict(count_dict)

def extract_patches_from_blobs(image_rgb, blobs, stride, window_size, enlarge_pixels=0):

    H, W, _ = image_rgb.shape
    patches = []

    for (y_hm, x_hm, r) in blobs:
        x_img = int(x_hm * stride)
        y_img = int(y_hm * stride)

        x1 = max(0, x_img)
        y1 = max(0, y_img)
        x2 = min(W, x1 + window_size)
        y2 = min(H, y1 + window_size)

        if x2 - x1 < window_size or y2 - y1 < window_size:
            continue 

        patch = image_rgb[y1-enlarge_pixels:y2+enlarge_pixels, x1-enlarge_pixels:x2+enlarge_pixels]
        patches.append(patch)

    return patches

def extract_features(patch, feature_list=None):
    """
    Extract selected features from a BGR patch.

    Parameters:
    - patch: Image patch in BGR format.
    - feature_list: List of feature names to extract. If None, extract all.

    Returns:
    - Concatenated feature vector (np.ndarray)
    """
    features = []
    if feature_list is None:
        feature_list = ['histogram', 'lbp', 'glcm', 'hog', 'color_moments', 'gabor']

    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

    if 'histogram' in feature_list:
        patch_hist = cv2.calcHist([patch], [0, 1, 2], None, [16, 16, 16], [0, 256]*3)
        patch_hist = cv2.normalize(patch_hist, patch_hist).flatten().astype(np.float32)
        features.append(patch_hist)

    if 'lbp' in feature_list:
        radius = 1
        n_points = 8 * radius
        lbp = local_binary_pattern(patch_gray, n_points, radius, method='uniform')
        lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        features.append(lbp_hist.astype(np.float32))

    if 'glcm' in feature_list:
        glcm = graycomatrix(patch_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)
        glcm_features = np.array([
            graycoprops(glcm, 'contrast').mean(),
            graycoprops(glcm, 'correlation').mean(),
            graycoprops(glcm, 'energy').mean(),
            graycoprops(glcm, 'homogeneity').mean()
        ], dtype=np.float32)
        features.append(glcm_features)

    if 'hog' in feature_list:
        hog_features = hog(patch_gray, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(1, 1), feature_vector=True)
        features.append(hog_features.astype(np.float32))

    if 'color_moments' in feature_list:
        color_moments = []
        for i in range(3):  # BGR channels
            channel = patch[:, :, i]
            mean = np.mean(channel)
            std = np.std(channel)
            skewness = skew(channel.reshape(-1))
            color_moments.extend([mean, std, skewness])
        features.append(np.array(color_moments, dtype=np.float32))

    if 'gabor' in feature_list:
        gabor_features = []
        frequencies = [0.1, 0.2, 0.3, 0.4]
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        for theta, freq in itertools.product(thetas, frequencies):
            kernel = cv2.getGaborKernel(ksize=(11, 11), sigma=4.0, theta=theta,
                                        lambd=1.0/freq, gamma=0.5, psi=0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(patch_gray, cv2.CV_8UC3, kernel)
            gabor_features.extend([np.mean(filtered), np.std(filtered)])
        features.append(np.array(gabor_features, dtype=np.float32))

    return np.concatenate(features, axis=0)

def extract_masked_features(image_bgr, mask, feature_list=None):

    features = []
    if feature_list is None:
        feature_list = ['histogram', 'lbp', 'glcm', 'hog', 'color_moments', 'gabor']

    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(image_gray, image_gray, mask=mask)
    masked_bgr = cv2.bitwise_and(image_bgr, image_bgr, mask=mask)

    if 'histogram' in feature_list:
        hist = cv2.calcHist([image_bgr], [0, 1, 2], mask, [16, 16, 16], [0, 256]*3)
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        features.append(hist)

    if 'lbp' in feature_list:
        radius, n_points = 1, 8
        lbp = local_binary_pattern(masked_gray, n_points, radius, method='uniform')
        lbp_masked = lbp[mask > 0]
        lbp_hist, _ = np.histogram(lbp_masked.ravel(), bins=np.arange(0, n_points + 3), density=True)
        features.append(lbp_hist.astype(np.float32))

    if 'glcm' in feature_list:
        normed_gray = cv2.normalize(masked_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        glcm = graycomatrix(normed_gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                            levels=256, symmetric=True, normed=True)
        glcm_features = [
            graycoprops(glcm, prop).mean()
            for prop in ['contrast', 'correlation', 'energy', 'homogeneity']
        ]
        features.append(np.array(glcm_features, dtype=np.float32))

    if 'hog' in feature_list:
        hog_features = hog(masked_gray, orientations=8, pixels_per_cell=(8, 8),
                           cells_per_block=(1, 1), feature_vector=True)
        features.append(hog_features.astype(np.float32))

    if 'color_moments' in feature_list:
        color_moments = []
        for i in range(3):  # BGR channels
            ch = masked_bgr[:, :, i][mask > 0]
            if len(ch) > 0:
                m, s, sk = np.mean(ch), np.std(ch), skew(ch)
            else:
                m, s, sk = 0.0, 0.0, 0.0
            color_moments.extend([m, s, sk])
        features.append(np.array(color_moments, dtype=np.float32))

    if 'gabor' in feature_list:
        gabor_features = []
        frequencies = [0.1, 0.2, 0.3, 0.4]
        thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        for theta, freq in itertools.product(thetas, frequencies):
            kernel = cv2.getGaborKernel(ksize=(11, 11), sigma=4.0, theta=theta,
                                        lambd=1.0/freq, gamma=0.5, psi=0, ktype=cv2.CV_32F)
            filtered = cv2.filter2D(masked_gray, cv2.CV_8UC3, kernel)
            masked_filtered = filtered[mask > 0]
            gabor_features.extend([np.mean(masked_filtered), np.std(masked_filtered)])
        features.append(np.array(gabor_features, dtype=np.float32))

    return np.concatenate(features, axis=0)
