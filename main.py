"""
深度学习视频换脸主程序
  - InsightFace inswapper（与 roop / FaceFusion 同类）
  - 可选 GFPGAN / CodeFormer 人脸修复
配置见 config.py；旧版 dlib 流程见 main_legacy.py
"""
import os
import cv2
from tqdm import tqdm

import config as cfg
from PythonCv2 import video2img, get_video_fps
from newvideo import mergevideo
from deletecache import del_file
from dl_swap import InsightFaceSwapper
from dl_restore import create_restorer


def _ensure_dirs():
    os.makedirs(cfg.ORIGIN_DIR, exist_ok=True)
    os.makedirs(cfg.TRANSFER_DIR, exist_ok=True)
    os.makedirs(cfg.MODELS_DIR, exist_ok=True)


def _sorted_frames(folder: str, ext: str) -> list[str]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(ext)]
    return sorted(files, key=lambda x: int(os.path.splitext(x)[0]))


def _imwrite(path: str, img, ext: str, quality: int):
    if ext.lower() in (".jpg", ".jpeg"):
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    else:
        cv2.imwrite(path, img)


def run():
    _ensure_dirs()

    if not os.path.isfile(cfg.SOURCE_FACE):
        raise FileNotFoundError(f"源人脸图片不存在: {cfg.SOURCE_FACE}")
    if not os.path.isfile(cfg.VIDEO_INPUT):
        raise FileNotFoundError(f"输入视频不存在: {cfg.VIDEO_INPUT}")

    fps = cfg.OUTPUT_FPS or get_video_fps(cfg.VIDEO_INPUT)
    ext = cfg.FRAME_EXT

    print("=" * 50)
    print("InsightFace 换脸 + 人脸修复")
    print(f"  视频: {cfg.VIDEO_INPUT}")
    print(f"  源脸: {cfg.SOURCE_FACE}")
    print(f"  修复: {cfg.RESTORE_METHOD}")
    print("=" * 50)

    print("\n[1/4] 拆分视频帧...")
    frame_count = video2img(
        cfg.VIDEO_INPUT, cfg.ORIGIN_DIR, ext=ext, jpeg_quality=cfg.JPEG_QUALITY
    )
    print(f"      共 {frame_count} 帧")

    print("\n[2/4] 加载 InsightFace 模型...")
    swapper = InsightFaceSwapper(
        models_dir=cfg.MODELS_DIR,
        face_analyser=cfg.FACE_ANALYSER,
        swapper_name=cfg.SWAPPER_MODEL,
        det_size=cfg.DET_SIZE,
        device=cfg.DEVICE,
    )
    swapper.set_source(cfg.SOURCE_FACE)
    print("      源人脸已加载")

    restorer = None
    if cfg.RESTORE_METHOD.lower() not in ("none", "off", ""):
        print(f"\n[3/4] 加载 {cfg.RESTORE_METHOD} 修复模型...")
        kwargs = {"models_dir": cfg.MODELS_DIR, "device": cfg.DEVICE}
        if cfg.RESTORE_METHOD.lower() == "codeformer":
            kwargs["fidelity"] = cfg.CODEFORMER_FIDELITY
        elif cfg.RESTORE_METHOD.lower() == "gfpgan":
            kwargs["arch"] = cfg.GFPGAN_ARCH
        restorer = create_restorer(cfg.RESTORE_METHOD, **kwargs)
        print("      修复模型已加载")
    else:
        print("\n[3/4] 跳过人脸修复")

    print("\n[4/4] 逐帧换脸" + (" + 修复" if restorer else "") + "...")
    frames = _sorted_frames(cfg.ORIGIN_DIR, ext)
    for i, name in enumerate(tqdm(frames, unit="帧")):
        frame_path = os.path.join(cfg.ORIGIN_DIR, name)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        result = swapper.swap_frame(frame)
        if restorer is not None:
            result = restorer.restore(result)

        out_name = os.path.splitext(name)[0] + ext
        _imwrite(os.path.join(cfg.TRANSFER_DIR, out_name), result, ext, cfg.JPEG_QUALITY)

        if (i + 1) % 100 == 0:
            print(f"      已完成 {i + 1}/{len(frames)} 帧")

    print("\n合并视频...")
    mergevideo(cfg.TRANSFER_DIR, cfg.OUTPUT_VIDEO, fps=fps, ext=ext)
    print(f"输出: {cfg.OUTPUT_VIDEO}")

    if cfg.ASK_DELETE_CACHE:
        m = input("\n是否删除 origin/ transfer 缓存? (y/n): ").strip().lower()
        if m == "y":
            del_file(cfg.ORIGIN_DIR)
            del_file(cfg.TRANSFER_DIR)
            print("缓存已删除")
        else:
            print("缓存保留")

    print("\nDone!")


if __name__ == "__main__":
    run()
