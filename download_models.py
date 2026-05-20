"""下载 InsightFace inswapper 等模型（官方 GitHub 失败时走 HuggingFace 镜像）。"""
import os
import urllib.request

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

INSWAPPER_URLS = [
    "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
    "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
]


def download_inswapper(dest: str | None = None) -> str:
    dest = dest or os.path.join(MODELS_DIR, "inswapper_128.onnx")
    if os.path.isfile(dest) and os.path.getsize(dest) > 1_000_000:
        print(f"已存在: {dest}")
        return dest

    last_err = None
    for url in INSWAPPER_URLS:
        try:
            print(f"下载 inswapper: {url}")
            urllib.request.urlretrieve(url, dest)
            if os.path.getsize(dest) > 1_000_000:
                print(f"完成: {dest}")
                return dest
        except Exception as e:
            last_err = e
            print(f"  失败: {e}")

    raise RuntimeError(f"inswapper 下载失败: {last_err}")


if __name__ == "__main__":
    download_inswapper()
    print("buffalo_l 将在首次运行 main.py 时由 InsightFace 自动下载到 models/buffalo_l/")
