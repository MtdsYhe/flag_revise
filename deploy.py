import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 国内镜像

from huggingface_hub import snapshot_download
snapshot_download(repo_id="Qwen/Qwen3-4B", local_dir="./model/Qwen3-4B", local_dir_use_symlinks=False)