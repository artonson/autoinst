docker run --gpus all \
    -v /mnt/hdd/KITTI_ODOMETRY/sequences/00/velodyne/:/input \
    -v /mnt/hdd/KITTI_ODOMETRY/sequences/00/velodyne_out/:/output \
    tarl