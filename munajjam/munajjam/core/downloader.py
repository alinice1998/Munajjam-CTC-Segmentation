import os
import requests
from pathlib import Path
from typing import Optional

def download_file(url: str, dest_path: str | Path, expected_size: Optional[int] = None):
    """
    Download a file from a URL with a progress bar using basic standard libraries.
    """
    dest_path = Path(dest_path)
    if dest_path.exists():
        if expected_size is None or dest_path.stat().st_size == expected_size:
            print(f"File {dest_path.name} already exists. Skipping download.")
            return

    print(f"Downloading {dest_path.name} from {url}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1 MB
    downloaded = 0
    
    with open(dest_path, 'wb') as f:
        for data in response.iter_content(block_size):
            f.write(data)
            downloaded += len(data)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}% ({downloaded / 1024 / 1024:.1f} MB)", end="")
    print(f"\nSuccessfully downloaded {dest_path.name}")

def ensure_fastconformer_model(model_dir: str | Path = "models", use_q8: bool = True) -> tuple[str, str]:
    """
    Ensures the FastConformer ONNX model and tokens are downloaded.
    Returns the paths to the (model_path, tokens_path).
    """
    model_dir = Path(model_dir)
    
    base_url = "https://github.com/Iam-Muslim/QuranReciteToText/releases/download/model"
    
    model_name = "qurankarim-fastconformer-q8.onnx" if use_q8 else "qurankarim-fastconformer-mixed.onnx"
    tokens_name = "tokens.txt"
    
    model_path = model_dir / model_name
    tokens_path = model_dir / tokens_name
    
    download_file(f"{base_url}/{model_name}", model_path)
    download_file(f"{base_url}/{tokens_name}", tokens_path)
    
    return str(model_path.absolute()), str(tokens_path.absolute())
