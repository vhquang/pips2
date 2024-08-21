import os
import time
import typing
from collections import namedtuple
from contextlib import contextmanager

import cv2
import numpy as np


def read_video(fp: str) -> typing.Iterable[np.ndarray]:
    """
    Iterate through frames in the video. Press 'q' to exit.

    Returns:
        - Iterable frame of [H, W, C]
    """
    cap = cv2.VideoCapture(fp)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            yield frame
    finally:
        cap.release()


def get_video_mask(fp: str, debug: bool = False):
    
    def absdiff_mask(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
        """
        https://stackoverflow.com/q/56183201/793367

        Returns:
            mask: [H x W] np.ndarray of bool
        """
        diff = cv2.absdiff(prev_frame, frame)                # [H, W, C]
        hsv_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)    # [H, W]
        
        def global_mask()-> tuple[int, np.ndarray]: 
            # thres, thres_mask = cv2.threshold(hsv_gray, 0, 255, type=cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            thres, thres_mask = cv2.threshold(hsv_gray, 30, 255, type=cv2.THRESH_BINARY)
            return thres, thres_mask
        
        def log_mask() -> tuple[int, np.ndarray]:
            log_scale = np.zeros(hsv_gray.shape).astype(np.float16)
            np.log(hsv_gray, out=log_scale, where=hsv_gray != 0)
            cutoff = log_scale.max() / 2
            log_scale = log_scale.astype(np.uint8)
            thres, thres_mask = cv2.threshold(log_scale, cutoff, 255, type=cv2.THRESH_BINARY)
            return thres, thres_mask

        def exp_mask() -> tuple[int, np.ndarray]:
            scaled = np.exp(hsv_gray / hsv_gray.max()) - 1
            norm = (scaled / scaled.max() * 255).astype(np.uint8)
            thres, thres_mask = cv2.threshold(norm, 40, 255, type=cv2.THRESH_BINARY)
            return thres, thres_mask
        
        mask, thres_mask = log_mask()
        mask = thres_mask == 255     # [H, W] of bool
        # cv2.imshow('Grayscale', hsv_gray)
        cv2.imshow('thres_mask', thres_mask)
        return mask

    def overlay(
        frame: np.ndarray, 
        mask: np.ndarray, 
        color: tuple[int, int, int] = (0, 0, 255), 
        opacity: float = 0.5,
    ) -> np.ndarray:
        """
        Args:
            frame: [H, W, C] int8
            mask: [H, W] bool
            color: BGR tuple
        """
        render = frame.copy()
        overlay_color = np.array(color) * opacity
        render[mask] = render[mask] * (1 - opacity) + overlay_color
        
        return render.astype(frame.dtype)
    
    # Calculate RMS of a frame
    def frame_rms(frame: np.ndarray) -> float:
        H, W, C = frame.shape
        pixels = frame.reshape((H * W, C))
        pixels_rms = (pixels ** 2).mean(axis=-1) ** 0.5  # RMS of BGR of each pixel
        res = (pixels_rms ** 2).mean() ** 0.5  # RMS of entire frame
        return res

    def bgr_rms(frame: np.ndarray) -> tuple[float, float, float]:
        H, W, C = frame.shape
        pixels = frame.reshape((H * W, C))
        B, G, R = (pixels ** 2).mean(axis=0) ** 0.5  
        return B, G, R

    reader = read_video(fp)
    prev_frame: np.ndarray = None
    mask: np.ndarray = None

    Col = namedtuple('Col', ['idx', 'mask_sum', 'rms', 'b', 'g', 'r'])
    plot_data: list[Col] = []

    
    for i, frame in enumerate(reader):
        if i == 0:
            prev_frame = frame
            mask = np.zeros(frame.shape[0:2]).astype(bool)
            continue

        # print(prev_frame.shape, frame.shape)
        diff_mask = absdiff_mask(prev_frame, frame)
        mask = np.logical_or(mask, diff_mask)

        # mask = diff_mask
        # print(f'{i=:02} {mask.sum():,}')
        # b, g, r = bgr_rms(frame)
        # rms = frame_rms(frame)
        # plot_data.append(Col(i, diff_mask.sum(), rms, b, g, r))

        render = overlay(frame, mask, opacity=0.5)
        
        if debug: cv2.imshow('Video with Vectors', render)
        prev_frame = frame.copy()
    
    cv2.destroyAllWindows()
    return mask