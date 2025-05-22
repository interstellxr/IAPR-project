import os
import cv2
import numpy as np
import pandas as pd
import skimage as ski
import matplotlib.pyplot as plt

from .utils import *
from tqdm import tqdm
import pickle

from typing import Union


class ChocoDetector:
    def __init__(
        self,
        sliding_window_size: int,
        sliding_window_stride: int,
        n_bins_histogram: int,
        heatmap_percentile: float,
        blob_min_sigma: float,
        blob_max_sigma: float,
        blob_thr: float,
        blob_avg_R: float,
        classifier_checkpoint_path: str,
        submission_template_df: pd.DataFrame,
        train_labels_df: pd.DataFrame = None,
        optimisation_mode: bool = False,
    ):
        self.sliding_window_size = sliding_window_size
        self.sliding_window_stride = sliding_window_stride
        channels_histogram = [0, 1, 2]
        self.hist_params = dict(
            channels=channels_histogram,
            histSize=[n_bins_histogram] * len(channels_histogram),
            ranges=[0, 256] * len(channels_histogram),
        )
        self.heatmap_percentile = heatmap_percentile
        self.blob_min_sigma = blob_min_sigma
        self.blob_max_sigma = blob_max_sigma
        self.blob_thr = blob_thr
        self.blob_avg_R = blob_avg_R
        with open(classifier_checkpoint_path, "rb") as f:
            self.classifier = pickle.load(f)

        self.column_names_map = {
            "jelly white": "Jelly White",
            "jelly milk": "Jelly Milk",
            "jelly black": "Jelly Black",
            "amandina": "Amandina",
            "creme brulee": "Crème brulée",
            "triangolo": "Triangolo",
            "tentation noir": "Tentation noir",
            "comtesse": "Comtesse",
            "noblesse": "Noblesse",
            "noir authentique": "Noir authentique",
            "passion au lait": "Passion au lait",
            "arabia": "Arabia",
            "stracciatella": "Stracciatella",
        }
        self.optim = optimisation_mode
        self.submission_template_df = submission_template_df.set_index("id")
        self.train_labels_df = train_labels_df.set_index("id")

    def load_data(
        self,
        ref_images_rgb: list[np.ndarray],
        ref_filenames: list[str],
        train_images_rgb: list[np.ndarray],
        train_filenames: list[str],
        test_images_rgb: list[np.ndarray],
        test_filenames: list[str],
    ):
        self.ref_images_rgb = ref_images_rgb
        self.ref_names = ref_filenames
        self.train_images_rgb = train_images_rgb
        self.train_names = train_filenames
        self.test_images_rgb = test_images_rgb
        self.test_names = test_filenames
        self.data_loaded = True

    def prepare_references(self):
        ref_images_gray = [
            cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in self.ref_images_rgb
        ]
        ref_masks = [
            compute_mask(img_gray=img, name=name)
            for img, name in zip(ref_images_gray, self.ref_names)
        ]
        self.ref_histograms = [
            compute_histogram(
                image=img, mask=mask, kernel_size=11, hist_params=self.hist_params
            )
            for img, mask in zip(self.ref_images_rgb, ref_masks)
        ]

    def compute_optim_score(self, prediction_df: pd.DataFrame) -> float:
        y_true = self.train_labels_df.copy(deep=True)
        y_pred = prediction_df.copy(deep=True)

        # only class columns
        Yt = y_true.to_numpy(dtype=np.int64)
        Yp = y_pred.to_numpy(dtype=np.int64)

        # True positives per image
        TP = np.minimum(Yt, Yp).sum(axis=1)  # shape (N,)
        # False-positive / negative penalty
        FPN = np.abs(Yt - Yp).sum(axis=1)

        denom = 2 * TP + FPN
        with np.errstate(divide="ignore", invalid="ignore"):
            F1_per_image = np.where(denom > 0, 2 * TP / denom, 0.0)

        return float(F1_per_image.mean())

    def run(self) -> Union[pd.DataFrame, float]:
        if not self.data_loaded:
            raise ValueError("Data has not been loaded. Please call load_data() first.")

        if self.optim:
            self.images = self.train_images_rgb
            self.names = self.train_names
            self.IDs = np.array(
                [int(filename[1:].split(".")[0]) for filename in self.names]
            )

            if self.train_labels_df is None:
                raise ValueError("In optimisation mode, training labels are necessary")

            self.prediction_df = self.train_labels_df.copy(deep=True)
            for id in self.IDs:
                self.prediction_df.loc[id, :] = 0
        else:
            self.images = self.test_images_rgb
            self.names = self.test_names
            self.IDs = np.array(
                [int(filename[1:].split(".")[0]) for filename in self.names]
            )
            self.prediction_df = self.submission_template_df.copy(deep=True)

        print(f"DEBUG - Optimisation is {self.optim}: {self.prediction_df.shape}")
        print(f"DEBUG - Running pipeline on {len(self.IDs)} samples")

        # 0) Compute and store reference histograms
        self.prepare_references()

        # 1) Compute heatmaps
        self.heatmaps = []
        for img in tqdm(self.images, unit="img"):
            heatmap = sliding_window_compare(
                img,
                self.ref_histograms,
                window_size=self.sliding_window_size,
                stride=self.sliding_window_stride,
                hist_params=self.hist_params,
            )
            self.heatmaps.append(heatmap)

        # 2) Filter heatmaps
        self.filtered_heatmaps = []
        for h in self.heatmaps:
            t = np.percentile(h, self.heatmap_percentile)
            self.filtered_heatmaps.append(np.where(h >= t, h, 0))

        # 3) Compute blobs
        self.blobs = [
            compute_blobs(
                fhm,
                # min_sigma=1.5,
                min_sigma=self.blob_min_sigma,
                # max_sigma=4,
                max_sigma=self.blob_max_sigma,
                # thr=0.06,
                thr=self.blob_thr,
                # avg_R=avg_R_px,
                avg_R=self.blob_avg_R,
                merge_policy="mean",
            )
            for fhm in self.filtered_heatmaps
        ]

        # 4) Extract crops
        self.all_crops = extract_crops(
            self.images,
            blobs=self.blobs,
            crop_size=64,
            window_size=self.sliding_window_size,
            stride=self.sliding_window_stride,
        )

        # 5) Extract crop features
        self.crops_features = [
            extract_crop_features(crop_list) for crop_list in self.all_crops
        ]

        # 6) Predict
        for id, crops_features in zip(self.IDs, self.crops_features):
            preds = self.classifier.predict(crops_features)
            for prediction in preds:
                if prediction != "ood":
                    self.prediction_df.loc[id, self.column_names_map[prediction]] += 1

        if self.optim:
            return self.compute_optim_score(prediction_df=self.prediction_df)
            # return self.prediction_df
        else:
            return self.prediction_df

    def show_sample_result(self):
        if self.optim:
            max = self.train_images_rgb.shape[0]
        else:
            max = self.test_images_rgb.shape[0]

        idx = np.random.randint(low=0, high=max)
        print(f"Image {idx}")
        plot_heatmap_result(
            filtered_heatmaps=self.filtered_heatmaps,
            images_rgb=self.images,
            blobs=self.blobs,
            window_size=self.sliding_window_size,
            stride=self.sliding_window_stride,
            idx=idx,
        )
