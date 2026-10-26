from torch.utils.data import Dataset
import os
import pyarrow.parquet as pq
from PIL import Image
from io import BytesIO
import torch

class ChestXrayDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = transform
        # Cache the currently open shard and row group
        self._cached_shard = None
        self._cached_pf = None
        self._cached_row_group = None
        self._cached_table = None
        self._cached_group_start = None
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        metadata = self.dataframe.iloc[idx]
        shard = metadata["shard"]
        # Open a shard only when it changes
        if shard != self._cached_shard:
            parquet_path = os.path.join(self.data_dir, shard)
            self._cached_pf = pq.ParquetFile(parquet_path)
            self._cached_shard = shard
            self._cached_row_group = None
        pf = self._cached_pf

        row_index = int(metadata["row_index"])
        group_start = 0
        # Find the row group containing this row       
        for g in range(pf.num_row_groups):
            group_rows = pf.metadata.row_group(g).num_rows
        
            if row_index < group_start + group_rows:
                row_group = g
                offset = row_index - group_start
                break
        
            group_start += group_rows
        else:
            raise IndexError(f"Row index {row_index} is out of range")
        # Read a row group only if it isn't already cached
        if row_group != self._cached_row_group:
            self._cached_table = pf.read_row_group(
                row_group, columns=["image", "labels"]
            )
            self._cached_row_group = row_group
            self._cached_group_start = group_start
        
        image_data = self._cached_table.slice(offset, 1)["image"][0].as_py()
        labels = self._cached_table.slice(offset, 1)["labels"][0].as_py()
        
        image = Image.open(BytesIO(image_data["bytes"])).convert("RGB")
        target = torch.zeros(14, dtype=torch.float32)

        for label_id in labels:
            if label_id != 0:
                target[label_id - 1] = 1.0
        
        if self.transform:
            image = self.transform(image)
        return image, target