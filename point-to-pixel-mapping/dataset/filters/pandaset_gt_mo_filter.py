from pandaset import DataSet, geometry
from scipy.spatial import Delaunay

from dataset.dataset import DatasetOrNone
from dataset.dataset_entry import DatasetEntry
from dataset.filters.filter import Filter
from dataset.types import DatasetPathLike


class PandasetGTMovingObjectFilter(Filter):
    def __init__(self, ds_path: DatasetPathLike, seq_num: int):
        """
        Pandaset Ground Truth Moving Object Filter
        Must be the first filter, otherwise shapes might mismatch

        Args:
            ds_path (DatasetPathLike): dataset path
            seq_num (int): seq num
        """
        self.dataset: DataSet = DataSet(ds_path)
        self.sequence_obj = self.dataset[str(seq_num).zfill(3)]
        self.sequence_obj.load_cuboids()  # takes some time
        self.cuboids = self.sequence_obj.cuboids

    def __call__(
        self, data_entry: DatasetEntry, dataset: DatasetOrNone = None
    ) -> DatasetEntry:
        points = data_entry.point_cloud

        cuboids = self.cuboids[data_entry.index]
        cuboids = cuboids.loc[cuboids["stationary"] == False]
        for _, row in cuboids.iterrows():
            hull = Delaunay(
                geometry.center_box_to_corners(
                    [
                        row["position.x"],
                        row["position.y"],
                        row["position.z"],
                        row["dimensions.x"],
                        row["dimensions.y"],
                        row["dimensions.z"],
                        row["yaw"],
                    ]
                )
            )
            points = points[hull.find_simplex(points) < 0]

        filtered_entry = DatasetEntry(
            data_entry.index, data_entry.pose, points, data_entry.images
        )

        return filtered_entry
