# Learning-Medical-Imaging

This is a great resource to brush-up python - https://cs231n.github.io/python-numpy-tutorial/

## Desktop Setup

### Step 1: Install CUDA-enabled PyTorch

Install PyTorch 2.6.0 and torchvision 0.21.0 with CUDA 11.8 support:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Verify GPU access

Run the following code in a Python script or Jupyter Notebook:

```python
import torch
import torchvision

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM (GB):",
        round(
            torch.cuda.get_device_properties(0).total_memory / 1024**3,
            2,
        ),
    )
```

CUDA availability should be `True`.

### Step 3: Download the dataset

Create the dataset directory relative to the project root:

```cmd
mkdir data\data
cd data\data
```

Download the six Parquet shards from Hugging Face using Windows Command Prompt or Anaconda Prompt:

```cmd
curl.exe --ssl-revoke-best-effort -L -C - -o train-00000-of-00006.parquet "https://huggingface.co/datasets/chehablab/NIHChestXR/resolve/main/data/train-00000-of-00006.parquet"

curl.exe --ssl-revoke-best-effort -L -C - -o train-00001-of-00006.parquet "https://huggingface.co/datasets/chehablab/NIHChestXR/resolve/main/data/train-00001-of-00006.parquet"

curl.exe --ssl-revoke-best-effort -L -C - -o train-00002-of-00006.parquet "https://huggingface.co/datasets/chehablab/NIHChestXR/resolve/main/data/train-00002-of-00006.parquet"

curl.exe --ssl-revoke-best-effort -L -C - -o train-00003-of-00006.parquet "https://huggingface.co/datasets/chehablab/NIHChestXR/resolve/main/data/train-00003-of-00006.parquet"

curl.exe --ssl-revoke-best-effort -L -C - -o train-00004-of-00006.parquet "https://huggingface.co/datasets/chehablab/NIHChestXR/resolve/main/data/train-00004-of-00006.parquet"

curl.exe --ssl-revoke-best-effort -L -C - -o train-00005-of-00006.parquet "https://huggingface.co/datasets/chehablab/NIHChestXR/resolve/main/data/train-00005-of-00006.parquet"
```
**Note:** Run the directory commands from the project root. If `data\data` already exists, you can skip the `mkdir` command.
