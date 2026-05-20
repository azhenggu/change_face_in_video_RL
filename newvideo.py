import os
import cv2
from PIL import Image


def _frame_sort_key(name: str) -> int:
    return int(os.path.splitext(name)[0])


def _list_frame_files(folder: str, ext: str) -> list[str]:
    """只保留数字命名的帧文件，如 0.jpg、1.jpg。"""
    ext = ext.lower()
    files = []
    for name in os.listdir(folder):
        if not name.lower().endswith(ext):
            continue
        stem = os.path.splitext(name)[0]
        if stem.isdigit():
            files.append(name)
    return sorted(files, key=_frame_sort_key)


def get_video_size(transferpath: str, ext: str = ".jpg") -> tuple[int, int]:
    files = _list_frame_files(transferpath, ext)
    if not files:
        raise FileNotFoundError(f"目录 {transferpath} 中没有 {ext} 帧图")
    img = Image.open(os.path.join(transferpath, files[0]))
    return img.size


def mergevideo(
    transferpath: str,
    output_path: str = "saveVideo.mp4",
    fps: float = 25,
    ext: str = ".jpg",
) -> None:
    """将 transfer 目录中的帧合成为视频（帧名从 0 开始：0.jpg, 1.jpg, ...）。"""
    img_root = transferpath
    files = _list_frame_files(transferpath, ext)
    if not files:
        raise FileNotFoundError(f"未找到帧图: {transferpath}")

    size = get_video_size(transferpath, ext)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, size)

    for name in files:
        frame = cv2.imread(os.path.join(img_root, name))
        if frame is None:
            continue
        if (frame.shape[1], frame.shape[0]) != size:
            frame = cv2.resize(frame, size)
        writer.write(frame)

    writer.release()
