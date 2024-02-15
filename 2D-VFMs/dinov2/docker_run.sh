#!/bin/bash

# Define the arrays for images and sequences
images=("image_2" "image_3")
sequences=("01" "02" "03" "04" "06" "07" "08" "09" "10")

# Loop over sequences
for seq in "${sequences[@]}"; do
    # Loop over images
    for img in "${images[@]}"; do
        docker run --gpus all \
            -v "/media/cedric/Datasets1/semantic_kitti/sequences/${seq}/${img}/:/input" \
            -v "/media/cedric/Datasets1/semantic_kitti/dinov2_features/${seq}/${img}/:/output" \
            dinov2 --facet 'query'
    done
done
