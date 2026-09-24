import math

import numpy as np

from shared.paths import LABEL
from shared.data import on_disk
from preprocess import prepare_many


class LesionBatches:
    def __init__(self, df, label_col=LABEL, batch_size=32, shuffle=True, seed=0, **options):
        self.label_col = label_col
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.options = options
        self.rng = np.random.default_rng(seed)

        self.df = on_disk(df.dropna(subset=[label_col]))
        self.classes = sorted(self.df[label_col].unique())
        self.class_index = {name: i for i, name in enumerate(self.classes)}
        self.skipped = []

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def _order(self):
        positions = np.arange(len(self.df))
        if self.shuffle:
            self.rng.shuffle(positions)
        return positions

    def __iter__(self):
        positions = self._order()
        for start in range(0, len(positions), self.batch_size):
            rows = self.df.iloc[positions[start:start + self.batch_size]]
            images, ids, failed = prepare_many(rows, **self.options)
            self.skipped.extend(failed)
            if not ids:
                continue
            names = rows.set_index("isic_id").loc[ids, self.label_col]
            labels = np.array([self.class_index[n] for n in names])
            yield images, labels, ids
