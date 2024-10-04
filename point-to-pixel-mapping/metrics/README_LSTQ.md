# Association score evaluator

In modified_LSTQ.py is a class `evaluator` that can be used to evaluate the association scores of a 3D instance segmentation. It's initialized with two parameters:
- `offset`: largest number of instances in a given scan, used to calculate intersection keys. Default for semanticKITTI is 2^32 because the gt labels are 32 bit integers.
- `min_points`: minimum number of points in a instance to be considered. Default is 30. Set to 0 to involve all instances.

The evaluator has following public variables:
- `preds`: list of dictionaries containing the predicted instances. The keys of the dictionaries are the instance ids and the values are the numbers of point (area) of the instances.
- `gts`: list of dictionaries containing the ground truth instances. The keys of the dictionaries are the instance ids and the values are the numbers of point (area) of the instances.
- `intersection`: dictionary containing the intersection of the predicted and ground truth instances. The keys of the dictionaries are calculated by: $$\text{intersect key} = \text{prediction id} + \text{gt id} * \text{offset}$$ and the values are the numbers of point (area) of the intersection.
- `S_assoc_list`: list of association scores for each point cloud. This list is only available after calling `get_eval()`.

The evaluator has following public methods:
- `reset()`: resets the evaluator to the initial state.
- `add_batch(pred_labels, gt_labels)`: adds a new prediction and ground truth to the evaluator. Both of the labels are a one dimensional array of the same length containing the instance ids of the points. This functuon can be called multiple times to add multiple point clouds, it updates the `gts`, `preds` and `intersection` variables.
- `get_eval()`: calculates the association scores for each point cloud and updates the `S_assoc_list`. The output of this function is the average association score over all point clouds.