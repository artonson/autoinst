### Copied from https://github.com/artonson/dense_lidar_recon ###

from typing import Union

from dataset.dataset import DatasetOrNone
from dataset.dataset_entry import DatasetEntry, DatasetEntryOrNone
from dataset.filters.filter import Filter
from dataset.filters.types import FilterTupleOrList


class FilterList(Filter):
    def __init__(self, filters: FilterTupleOrList) -> None:
        self.filters = filters

    def __call__(
        self, data_entry: DatasetEntry, dataset: DatasetOrNone = None
    ) -> DatasetEntryOrNone:
        for filter in self.filters:
            data_entry = filter(data_entry, dataset=dataset)
        return data_entry


FilterListOrNone = Union[FilterList, None]
