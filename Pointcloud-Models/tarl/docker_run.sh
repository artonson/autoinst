#!/bin/bash
sequences=("00" "01" "02" "03" "05" "06" "07" "08" "09" "10")


# Loop over sequences
for seq in "${sequences[@]}"; do
    docker run --gpus all \
        -v "/media/cedric/Datasets2/semantic_kitti/sequences/${seq}/velodyne/:/input" \
        -v "/media/cedric/Datasets21/out/${seq}:/output" \
        tarl
done