sudo docker run --gpus all \
    -v /mnt/hdd/KITTI_ODOMETRY/sequences/00/image_2/:/input \
    -v /mnt/hdd/KITTI_ODOMETRY/sequences/test_docker/00/image_2/:/output \
    sam_test:v2 python run.py -f png #image format, default is png