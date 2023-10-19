import gc

import numpy as np

from segmentation.abstract_segmentation import AbstractSegmentation
from segmentation.utils.aggregate import AggregationClustering


class AggregatedSegmentation(AbstractSegmentation):
    def __init__(
            self,
            dataset,
            dataset_name='kitti',
            clusterer='dbscan',
            sequence='00',
            window=10):
        self.dataset = dataset
        self.window = window
        self.dataset_name = dataset_name

        if dataset_name == 'kitti':
            self.base_dir = self.dataset.dataset_path + \
                'sequences/' + self.dataset.sequence + '/'

            self.last_idx = len(self.dataset.poses) - 1

        elif dataset_name == 'nuScenes':
            self.last_idx = len(self.dataset.tokens['LIDAR_TOP']) - 1

        self.clustering = AggregationClustering(
            dataset_name=dataset_name,
            clusterer=clusterer,
            sequence=sequence,
            dataset=dataset)

    def segment_instances(self, index):
        points_set, ground_label, parse_idx = None, None, None
        segments = None

        start = index - self.window // 2
        end = index + self.window // 2 + 1
        if start < 0:
            diff = abs(start)
            end = end + diff
            start = 0
            aggregated_file_nums = list(range(start, end))
        elif end > self.last_idx:
            diff = end - self.last_idx
            start = start - diff
            end = self.last_idx + 1

            aggregated_file_nums = list(range(start, end))

        else:
            aggregated_file_nums = list(range(start, end))

        if self.dataset_name == 'kitti':
            fns = [
                self.base_dir +
                'velodyne/' +
                str(i).zfill(6) +
                '.bin' for i in aggregated_file_nums]
            points_set, ground_label, parse_idx = self.clustering.aggregate_pcds(
                fns, self.dataset.dataset_path)

        elif self.dataset_name == 'nuScenes':
            points_set, ground_label, parse_idx = self.clustering.aggregate_pcds_nuscenes(
                aggregated_file_nums)

        segments = self.clustering.clusterize_pcd(points_set, ground_label)
        segments[parse_idx] = -np.inf
        segments = segments.astype(np.float16)

        pcd_parse_idx = np.unique(np.argwhere(segments == -np.inf)[:, 0])
        # test extraction with one point cloud (the first one)
        test_idx = aggregated_file_nums.index(index)

        if self.dataset_name == 'kitti':
            pts = np.fromfile(
                fns[test_idx],
                dtype=np.float32)  # full points file
            pts = pts.reshape((-1, 4))
        elif self.dataset_name == 'nuScenes':
            pts = self.dataset.get_point_cloud(index)

        seg = segments[pcd_parse_idx[test_idx] +
                       1:pcd_parse_idx[test_idx + 1]]  # non ground points

        ps1 = np.concatenate((pts, seg), axis=-1)
        pts = ps1[:, :-1]  # remove the 5th dimension from point cloud tensor
        s1 = ps1[:, -1][:, np.newaxis]  # labels
        gc.collect()
        return s1.reshape(-1)
