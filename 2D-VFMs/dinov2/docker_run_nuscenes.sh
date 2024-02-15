#!/bin/bash

# Define the arrays for images and sequences
images=("CAM_FRONT_LEFT" "CAM_FRONT" "CAM_FRONT_RIGHT")

# Loop over images
for img in "${images[@]}"; do
        docker run --gpus all \
            -v "/media/cedric/Datasets1/nuScenes_train/${img}/:/input" \
            -v "/media/cedric/Datasets1/nuScenes_train/Dinov2/${img}/:/output" \
            dinov2 --facet 'query' -f '.jpg' -s 14 
done
