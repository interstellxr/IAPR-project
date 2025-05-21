import importlib
from utils import *
utils = importlib.import_module('utils')
importlib.reload(utils)

from skimage.feature import canny
from skimage.transform import hough_ellipse
from skimage.draw import ellipse_perimeter
from sklearn.decomposition import PCA
from skimage.feature import blob_log
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
import pandas as pd

from sklearn.preprocessing import StandardScaler

def compute_reference_features(ref_images_rgb: list, ref_images_gray: list, ref_image_names: list, features: list):
    masks = []
    for name, image in zip(ref_image_names, ref_images_gray):
        masks.append(find_mask(img_gray=image, name=name))

    overlap_ratio = 0.5
    bbox_coords = []
    bbox_sizes = []

    max_height = 0
    max_width = 0

    for i, mask in enumerate(masks):
        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        h = y_max - y_min
        w = x_max - x_min
        bbox_coords.append((y_min, y_max, x_min, x_max))
        bbox_sizes.append((h, w))

        max_height = max(max_height, h)
        max_width = max(max_width, w)

    min_height = min(h for h, w in bbox_sizes)
    min_width = min(w for h, w in bbox_sizes)
    min_size = (min_height // 2, min_width // 2)

    stride_y = int(min_size[0] * (1 - overlap_ratio))
    stride_x = int(min_size[1] * (1 - overlap_ratio))

    reference_feature_vectors = []  # list of list of patch feature vectors per image
    selected_bbox_coords = []

    for i in range(len(ref_images_rgb)):
        y_min, y_max, x_min, x_max = bbox_coords[i]
        mask = masks[i].astype(np.uint8) * 255

        reference_features = []

        for y in range(y_min, y_max - min_size[0] + 1, stride_y):
            for x in range(x_min, x_max - min_size[1] + 1, stride_x):
                tile_mask = mask[y:y+min_size[0], x:x+min_size[1]]
                coverage = np.sum(tile_mask) / (min_size[0] * min_size[1] * 255)

                if coverage >= 0.9:
                    roi = ref_images_rgb[i][y:y+min_size[0], x:x+min_size[1]]
                    roi_mask = tile_mask
                    feature_vec = extract_masked_features(roi, roi_mask, feature_list=features)
                    reference_features.append(feature_vec)
                    selected_bbox_coords.append((i, y, y + min_size[0], x, x + min_size[1]))

        # mean_features = np.mean(reference_features, axis=0)
        # reference_features.append(mean_features)

        reference_feature_vectors.append(reference_features)

    return ref_images_rgb, reference_feature_vectors, masks, min_size, selected_bbox_coords

def pca_on_reference_features(reference_feature_vectors: list, pca_percentile: float):

    ## Standardize the features
    patch_counts = [len(patches) for patches in reference_feature_vectors]
    flat_features = np.vstack(reference_feature_vectors) 

    # Standardize the features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(flat_features)

    ## PCA on reference features
    pca = PCA(n_components=pca_percentile)
    # feature_array = np.vstack(reference_feature_vectors)
    # new_array = pca.fit_transform(feature_array)
    new_array = pca.fit_transform(standardized_features)

    reduced_feature_vectors = []
    start = 0
    for count in patch_counts:
        end = start + count
        reduced_feature_vectors.append(new_array[start:end])
        start = end

    return reduced_feature_vectors, pca, scaler

def compute_reference_histograms(ref_images_rgb: list, masks: list):

    ## Reference histograms used for sliding window
    reference_histograms = []
    for i, (img_rgb, mask) in enumerate(zip(ref_images_rgb, masks)):
        img_rgb = img_rgb.copy()
        mask_uint8 = (mask.astype(np.uint8)) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)

        hist = cv2.calcHist([img_rgb], [0, 1, 2], eroded_mask, [16, 16, 16], [0, 256]*3)
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        reference_histograms.append(hist)

    return reference_histograms

def load_training_images(train_path: str):

    ## Load training images
    train_image_names = os.listdir(train_path)
    train_images = [cv2.imread(os.path.join(train_path, train_image_names[i])) for i in range(len(train_image_names))]
    train_images = [cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4)) for img in train_images]
    train_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in train_images]
    train_images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in train_images]

    return train_images_rgb, train_images_gray, train_image_names

def sliding_window(train_images_rgb: list, reference_histograms: list, window_size: int, step_size: int, blob_percentile: float):

    ## Perform sliding window detection
    heatmaps= []
    patch_histograms = []
    for img in train_images_rgb:
        heatmap, patch_hist = sliding_window_compare(img, reference_histograms, window_size=window_size, stride=step_size)
        threshold = np.percentile(heatmap, blob_percentile)
        heatmap = np.where(heatmap >= threshold, heatmap, 0)
        heatmaps.append(heatmap)
        patch_histograms.append(patch_hist)

    ## Find blobs in heatmaps
    blobs = []
    for heatmap in heatmaps:
        blobs.append(blob_log(heatmap, min_sigma=2, max_sigma=15, threshold=0.1))

    merged_blobs_all = []
    for blob_list in blobs:
        if len(blob_list) == 0:
            merged_blobs_all.append([])
            continue

        positions = blob_list[:, :2]

        clustering = DBSCAN(eps=step_size, min_samples=1).fit(positions)

        merged = []
        for label in np.unique(clustering.labels_):
            members = blob_list[clustering.labels_ == label]
            y_mean = np.mean(members[:, 0])
            x_mean = np.mean(members[:, 1])
            r_max = np.max(members[:, 2]) * np.sqrt(2) 
            merged.append((y_mean, x_mean, r_max))
        
        merged_blobs_all.append(merged)

    blobs = merged_blobs_all

    return blobs, patch_histograms

def extract_patch_features(train_images_rgb: list, blobs: list, step_size: int, window_size: int, features: list, min_size: tuple):

    ## Exract patches around blobs, resize and compute features
    patches_ = []
    for img, blob in zip(train_images_rgb, blobs):
        patches_.append(extract_patches_from_blobs(img, blob, stride=step_size, window_size=window_size, enlarge_pixels=0))

    resized_patches = []
    for patch in patches_:
        resized_patch = []
        for p in patch:
            rp = cv2.resize(p, (min_size[1], min_size[0])) 
            resized_patch.append(rp)
        resized_patches.append(resized_patch)

    patch_features = []
    for i, patch in enumerate(resized_patches):
        feature = []
        for p in patch:
            feature.append(extract_features(p, feature_list=features))
        patch_features.append(feature)

    ## Map the patches to their corresponding images
    patch_image_indices = []

    for image_idx, patches in enumerate(patch_features):
        n_patches = len(patches)
        patch_image_indices.extend([image_idx] * n_patches)

    patch_image_indices = np.array(patch_image_indices)

    return patch_features, patch_image_indices

def pca_on_patch_features(patch_features: list, pca: PCA, scaler: StandardScaler):

    ## Standardize the features
    patch_counts = [len(patches) for patches in patch_features]
    flat_features = np.vstack(patch_features)
    standardized_features = scaler.transform(flat_features)

    ## PCA on patch features
    # patch_feature_array = np.vstack(patch_features)
    patch_array = pca.transform(standardized_features)

    reduced_features = []
    start = 0
    for count in patch_counts:
        end = start + count
        reduced_features.append(patch_array[start:end])
        start = end

    return reduced_features

def classify_patches(reference_array: np.ndarray, train_array: np.ndarray, selected_bbox_coords: list, patch_image_indices: list, ood_percentile: float,
                     patch_features: list, train_image_names: list, df_gt: pd.DataFrame):

    ## Classification using KMeans
    k = 13
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(reference_array)

    patch_origins = np.array([item[0] for item in selected_bbox_coords])

    training_cluster_labels = kmeans.predict(train_array)

    ## OOD analysis
    distances = kmeans.transform(train_array)
    min_distances = distances.min(axis=1)

    reference_distances = kmeans.transform(reference_array).min(axis=1)
    threshold = np.percentile(min_distances, ood_percentile)

    is_in_distribution = min_distances < threshold

    patch_counts_training = [len(patches) for patches in patch_features]

    filtered_labels_per_image = []
    start = 0
    for count in patch_counts_training:
        end = start + count
        labels = training_cluster_labels[start:end]
        in_dist = is_in_distribution[start:end]
        
        filtered_labels = labels[in_dist]
        filtered_labels_per_image.append(filtered_labels)
        
        start = end

    ## Prepare the results
    new_train_names = [name.split('L')[1].split('.')[0] for name in train_image_names]
    new_ref_names = [
        'Passion au lait',
        'Noblesse',
        'Noir authentique',
        'Amandina',
        'Stracciatella',
        'Jelly White',
        'Arabia',
        'Triangolo',
        'Crème brulée',
        'Jelly Black',
        'Tentation noir',
        'Jelly Milk',
        'Comtesse'
        ]
    
    image_cluster_counts = defaultdict(lambda: [0] * k)

    for i, label in enumerate(training_cluster_labels):
        if is_in_distribution[i]:
            image_index = patch_image_indices[i]
            image_id = new_train_names[image_index] 
            image_cluster_counts[image_id][label] += 1
        else:
            image_index = patch_image_indices[i]
            image_id = new_train_names[image_index] 
            image_cluster_counts[image_id][label] += 0

    sorted_ids = sorted(image_cluster_counts.keys())

    chocolate_classes = new_ref_names 

    df_pred = pd.DataFrame.from_dict(image_cluster_counts, orient='index', columns=chocolate_classes)
    df_pred = df_pred.reset_index().rename(columns={'index': 'id'}) 
    df_pred = df_pred.sort_values('id').reset_index(drop=True) 
    df_pred = df_pred[df_gt.columns] 

    return df_pred

def ChocoDetectorv2(ref_images_rgb: list, ref_images_gray: list, ref_image_names: list, rgb_images: list, train_names: list, features: list, pca_percentile: float,
                  blob_percentile: float, window_size: int, step_size: int, ood_percentile: float, 
                  df_gt: pd.DataFrame):
    

    ref_images_rgb, reference_features, masks, min_size, bbox_coords = compute_reference_features(ref_images_rgb, ref_images_gray, ref_image_names, features=features)
    
    reference_histograms = compute_reference_histograms(ref_images_rgb, masks)

    reference_array = np.vstack(reference_features)

    reduced_reference_features, pca, scaler = pca_on_reference_features(reference_features, pca_percentile)
    reduced_reference_array = np.vstack(reduced_reference_features)

    blobs, patch_histograms = sliding_window(rgb_images, reference_histograms, window_size=window_size, step_size=step_size, blob_percentile=blob_percentile)

    patch_features, patch_image_indices = extract_patch_features(rgb_images, blobs, step_size=step_size, window_size=window_size, features=features, min_size=min_size)

    reduced_patch_features = pca_on_patch_features(patch_features, pca, scaler)
    reduced_patch_array = np.vstack(reduced_patch_features)

    df_pred = classify_patches(reduced_reference_array, reduced_patch_array, bbox_coords, patch_image_indices, ood_percentile=95, patch_features=patch_features, train_image_names=train_names, df_gt=df_gt)

    return df_pred

def ChocoDetector(ref_path: str, train_path: str, features: list, pca_percentile: float,
                  blob_percentile: float, window_size: int, step_size: int, ood_percentile: float, 
                  dt_gt: pd.DataFrame):
    
    ## Load reference images
    ref_image_names = os.listdir(ref_path)
    ref_images = [cv2.imread(os.path.join(ref_path, img)) for img in ref_image_names]
    ref_images = [cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4)) for img in ref_images]
    ref_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in ref_images]
    ref_images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in ref_images]

    ## Segment reference images and compute features
    masks = []
    for name, image in zip(ref_image_names, ref_images_gray):
        masks.append(find_mask(img_gray=image, name=name))

    overlap_ratio = 0.5  # 50% overlap

    bbox_coords = []
    bbox_sizes = []

    max_height = 0
    max_width = 0

    for i, mask in enumerate(masks):
        ys, xs = np.where(mask)
        y_min, y_max = ys.min(), ys.max()
        x_min, x_max = xs.min(), xs.max()

        h = y_max - y_min
        w = x_max - x_min
        bbox_coords.append((y_min, y_max, x_min, x_max))
        bbox_sizes.append((h, w))

        max_height = max(max_height, h)
        max_width = max(max_width, w)

    min_height = min(h for h, w in bbox_sizes)
    min_width = min(w for h, w in bbox_sizes)
    min_size = (min_height // 2, min_width // 2)

    stride_y = int(min_size[0] * (1 - overlap_ratio))
    stride_x = int(min_size[1] * (1 - overlap_ratio))

    reference_feature_vectors = []
    selected_bbox_coords = []

    for i in range(len(ref_images)):
        y_min, y_max, x_min, x_max = bbox_coords[i]
        mask = masks[i].astype(np.uint8) * 255

        reference_features = []

        for y in range(y_min, y_max - min_size[0] + 1, stride_y):
            for x in range(x_min, x_max - min_size[1] + 1, stride_x):
                tile_mask = mask[y:y+min_size[0], x:x+min_size[1]]
                coverage = np.sum(tile_mask) / (min_size[0] * min_size[1] * 255)

                if coverage >= 0.9:
                    roi = ref_images[i][y:y+min_size[0], x:x+min_size[1]]
                    roi_mask = tile_mask
                    feature_vec = extract_masked_features(roi, roi_mask, feature_list=features)
                    reference_features.append(feature_vec)
                    selected_bbox_coords.append((i, y, y + min_size[0], x, x + min_size[1]))

        reference_feature_vectors.append(reference_features)
    
    feature_array = np.vstack(reference_feature_vectors)

    ## PCA on reference features

    pca = PCA(n_components=pca_percentile)
    new_array = pca.fit_transform(feature_array)

    patch_counts = [len(patches) for patches in reference_feature_vectors]

    reduced_feature_vectors = []
    start = 0
    for count in patch_counts:
        end = start + count
        reduced_feature_vectors.append(new_array[start:end])
        start = end

    reduced_array = np.vstack(reduced_feature_vectors)

    ## Reference histograms used for sliding window
    reference_histograms = []
    for i, (img_rgb, mask) in enumerate(zip(ref_images_rgb, masks)):
        img_rgb = img_rgb.copy()
        mask_uint8 = (mask.astype(np.uint8)) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        eroded_mask = cv2.erode(mask_uint8, kernel, iterations=1)

        hist = cv2.calcHist([img_rgb], [0, 1, 2], eroded_mask, [16, 16, 16], [0, 256]*3)
        hist = cv2.normalize(hist, hist).flatten()
        reference_histograms.append(hist)

    ## Load training images
    train_image_names = os.listdir(train_path)
    train_images = [cv2.imread(os.path.join(train_path, train_image_names[i])) for i in range(len(train_image_names))]
    train_images = [cv2.resize(img, (img.shape[1] // 4, img.shape[0] // 4)) for img in train_images]
    train_images_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in train_images]
    train_images_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in train_images]

    ## Perform sliding window detection
    heatmaps= []
    for img in train_images_rgb:
        heatmap = sliding_window_compare(img, reference_histograms, window_size=window_size, stride=step_size)
        threshold = np.percentile(heatmap, blob_percentile)
        heatmap = np.where(heatmap >= threshold, heatmap, 0)
        heatmaps.append(heatmap)

    ## Find blobs in heatmaps
    blobs = []
    for heatmap in heatmaps:
        blobs.append(blob_log(heatmap, min_sigma=2, max_sigma=15, threshold=0.1))

    merged_blobs_all = []
    for blob_list in blobs:
        if len(blob_list) == 0:
            merged_blobs_all.append([])
            continue

        positions = blob_list[:, :2]

        clustering = DBSCAN(eps=step_size, min_samples=1).fit(positions)

        merged = []
        for label in np.unique(clustering.labels_):
            members = blob_list[clustering.labels_ == label]
            y_mean = np.mean(members[:, 0])
            x_mean = np.mean(members[:, 1])
            r_max = np.max(members[:, 2]) * np.sqrt(2) 
            merged.append((y_mean, x_mean, r_max))
        
        merged_blobs_all.append(merged)

    blobs = merged_blobs_all

    ## Exract patches around blobs, resize and compute features
    patches_ = []
    for img, blob in zip(train_images_rgb, blobs):
        patches_.append(extract_patches_from_blobs(img, blob, stride=step_size, window_size=window_size, enlarge_pixels=0))

    resized_patches = []
    for patch in patches_:
        resized_patch = []
        for p in patch:
            rp = cv2.resize(p, (min_size[1], min_size[0])) 
            resized_patch.append(rp)
        resized_patches.append(resized_patch)

    patch_features = []
    for i, patch in enumerate(resized_patches):
        feature = []
        for p in patch:
            feature.append(extract_features(p, feature_list=features))
        patch_features.append(feature)

    ## PCA on patch features
    patch_feature_array = np.vstack(patch_features)
    patch_array = pca.transform(patch_feature_array)
    patch_counts = [len(patches) for patches in patch_features]

    reduced_features = []
    start = 0
    for count in patch_counts:
        end = start + count
        reduced_features.append(patch_array[start:end])
        start = end

    train_array = np.vstack(reduced_features)

    ## Map the patches to their corresponding images
    patch_image_indices = []

    for image_idx, patches in enumerate(patch_features):
        n_patches = len(patches)
        patch_image_indices.extend([image_idx] * n_patches)

    patch_image_indices = np.array(patch_image_indices)

    ## Classification using KMeans
    k = 13
    kmeans = KMeans(n_clusters=k, random_state=0)
    cluster_labels = kmeans.fit_predict(reduced_array)

    patch_origins = np.array([item[0] for item in selected_bbox_coords])

    training_cluster_labels = kmeans.predict(train_array)

    ## OOD analysis
    distances = kmeans.transform(train_array)
    min_distances = distances.min(axis=1)

    reference_distances = kmeans.transform(reduced_array).min(axis=1)
    threshold = np.percentile(min_distances, ood_percentile)

    is_in_distribution = min_distances < threshold

    patch_counts_training = [len(patches) for patches in patch_features]

    filtered_labels_per_image = []
    start = 0
    for count in patch_counts_training:
        end = start + count
        labels = training_cluster_labels[start:end]
        in_dist = is_in_distribution[start:end]
        
        filtered_labels = labels[in_dist]
        filtered_labels_per_image.append(filtered_labels)
        
        start = end

    ## Prepare the results
    new_train_names = [name.split('L')[1].split('.')[0] for name in train_image_names]
    new_ref_names = [
        'Passion au lait',
        'Noblesse',
        'Noir authentique',
        'Amandina',
        'Stracciatella',
        'Jelly White',
        'Arabia',
        'Triangolo',
        'Crème brulée',
        'Jelly Black',
        'Tentation noir',
        'Jelly Milk',
        'Comtesse'
        ]
    
    image_cluster_counts = defaultdict(lambda: [0] * k)

    for i, label in enumerate(training_cluster_labels):
        if is_in_distribution[i]:
            image_index = patch_image_indices[i]
            image_id = new_train_names[image_index] 
            image_cluster_counts[image_id][label] += 1
        else:
            image_index = patch_image_indices[i]
            image_id = new_train_names[image_index] 
            image_cluster_counts[image_id][label] += 0

    sorted_ids = sorted(image_cluster_counts.keys())

    df_gt = pd.read_csv('chocolate-recognition-classic/dataset_project_iapr2025/train.csv')

    chocolate_classes = new_ref_names 

    df_pred = pd.DataFrame.from_dict(image_cluster_counts, orient='index', columns=chocolate_classes)
    df_pred = df_pred.reset_index().rename(columns={'index': 'id'}) 
    df_pred = df_pred.sort_values('id').reset_index(drop=True) 
    df_pred = df_pred[df_gt.columns] 

    ## Return the DataFrame with predictions
    return df_pred