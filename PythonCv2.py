#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2


def video2img(videofilepath: str, originpath: str, ext: str = ".jpg", jpeg_quality: int = 95) -> int:
    """将视频拆分为帧，保存为 0.jpg, 1.jpg, ... 返回帧数。"""
    os.makedirs(originpath, exist_ok=True)
    vc = cv2.VideoCapture(videofilepath)
    if not vc.isOpened():
        raise FileNotFoundError(f"无法打开视频: {videofilepath}")

    c = 0
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if ext.lower() in (".jpg", ".jpeg") else []

    while True:
        rval, frame = vc.read()
        if not rval:
            break
        out_path = os.path.join(originpath, f"{c}{ext}")
        if params:
            cv2.imwrite(out_path, frame, params)
        else:
            cv2.imwrite(out_path, frame)
        c += 1

    vc.release()
    return c


def get_video_fps(videofilepath: str, default: float = 25.0) -> float:
    vc = cv2.VideoCapture(videofilepath)
    fps = vc.get(cv2.CAP_PROP_FPS) or default
    vc.release()
    return fps if fps > 0 else default
