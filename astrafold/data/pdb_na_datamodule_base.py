"""
Lightning datamodule and the length-based batch sampler.

Code adapted from MMDiff (https://github.com/Profluent-Internships/MMDiff) and
https://github.com/microsoft/protein-frame-flow/blob/main/data/pdb_dataloader.py
"""

import math

import torch
from beartype.typing import Optional
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, dist

from astrafold.config.hydra_schema import DataConfig
from astrafold.data.pdb_na_dataset_base import PDBNABaseDataset


class PDBNABaseDataModule(LightningDataModule):
    def __init__(self, data_cfg: DataConfig):
        super().__init__()
        self.data_cfg = data_cfg
        self.data_train: Optional[PDBNABaseDataset] = None
        self.data_val: Optional[PDBNABaseDataset] = None

    def setup(self, stage):
        self.data_train = PDBNABaseDataset(self.data_cfg, is_training=True)
        if self.data_cfg.val_csv_path is not None:
            self.data_val = PDBNABaseDataset(
                self.data_cfg,
                is_training=False,
                csv_path=self.data_cfg.val_csv_path,
            )

    def train_dataloader(self, rank=None, num_replicas=None):
        num_workers = self.data_cfg.num_workers
        # Lightning's overfit_batches takes the first batch of every epoch, so the
        # batch order must not be reshuffled between epochs.
        overfit = self.trainer is not None and self.trainer.overfit_batches > 0
        batch_sampler = RNALengthBatcher(
            sampler_cfg=self.data_cfg,
            metadata_csv=self.data_train.csv,
            rank=rank,
            num_replicas=num_replicas,
            freeze_epoch=overfit,
        )
        return DataLoader(
            self.data_train,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else self.data_cfg.prefetch_factor,
            pin_memory=False,
            persistent_workers=num_workers > 0,
        )

    def val_dataloader(self):
        if self.data_val is None:
            return []

        # Batch size 1: validation RNAs have different lengths and are not grouped
        # into same-length batches like in training. Under DDP each rank evaluates its
        # own shard; DistributedSampler pads the shards to equal size by repeating up
        # to num_replicas - 1 examples.
        distributed = dist.is_available() and dist.is_initialized()
        val_sampler = (
            DistributedSampler(self.data_val, shuffle=False) if distributed else None
        )
        num_workers = min(self.data_cfg.num_workers, 2)
        return DataLoader(
            self.data_val,
            batch_size=1,
            sampler=val_sampler,
            shuffle=False,
            num_workers=num_workers,
            prefetch_factor=None if num_workers == 0 else 2,
            persistent_workers=num_workers > 0,
        )


class RNALengthBatcher:
    """
    Batch sampler that groups RNAs of the same length, so batches need no padding.

    Every replica (GPU) gets its own share of the data, and every replica yields the
    same number of batches per epoch: ceil(num_examples / num_replicas), independent
    of the batch size. The data is reshuffled and re-batched until that number is
    reached, so one "epoch" can cover the data more than once.
    """

    def __init__(
        self,
        sampler_cfg: DataConfig,
        metadata_csv,
        seed=123,
        shuffle=True,
        num_replicas=None,
        rank=None,
        freeze_epoch=False,
    ):
        super().__init__()
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        self._num_batches = math.ceil(len(self._data_csv) / self.num_replicas)
        self._data_csv["index"] = list(range(len(self._data_csv)))
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.freeze_epoch = freeze_epoch
        self.max_batch_size = self._sampler_cfg.max_batch_size

    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self._data_csv), generator=rng).tolist()
        else:
            indices = list(range(len(self._data_csv)))

        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[indices[self.rank :: self.num_replicas]]
        else:
            replica_csv = self._data_csv

        # Each batch contains multiple RNA of the same length. Longer RNAs get
        # smaller batches, to bound the memory of the pair representation.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby("modeled_na_seq_len"):
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // seq_len**2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i * max_batch_size : (i + 1) * max_batch_size]
                batch_indices = batch_df["index"].tolist()
                sample_order.append(batch_indices)

        # Remove any length bias.
        new_order = torch.randperm(len(sample_order), generator=rng).numpy().tolist()
        return [sample_order[i] for i in new_order]

    def _create_batches(self):
        # Make sure all replicas have the same number of batches. Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError("Exceeded number of augmentations.")
        self.sample_order = all_batches[: self._num_batches]

    def __iter__(self):
        self._create_batches()
        if not self.freeze_epoch:
            self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return self._num_batches
