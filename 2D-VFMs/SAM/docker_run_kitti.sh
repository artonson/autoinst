#!/bin/bash


#!/bin/bash

# Define the arrays for images and sequences
images=("image_2" "image_3")
#sequences=("01" "02" "03" "04" "06" "07" "08" "09" "10")
sequences=("00")



# Loop over images
for seq in "${sequences[@]}"; do
    for img in "${images[@]}"; do
        docker run --gpus all  \
            -v "/media/cedric/Datasets2/semantic_kitti/sequences/${seq}/${img}/:/input" \
            -v "/media/cedric/Datasets2/semantic_kitti/outputs/${seq}/${img}/:/output" \
            sam  
    done 
done

