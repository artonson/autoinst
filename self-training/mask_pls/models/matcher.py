# Modified by Rodrigo Marcuzzi from https://github.com/facebookresearch/Mask2Former
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment as lsa
from torch import nn
from torch.cuda.amp import autocast


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, costs: list, p_ratio,freepoint_loss):
        """Creates the matcher

        Params:
            weight_class: This is the relative weight of the classification error in the matching cost
            weight_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
            weight_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
        """
        super().__init__()
        self.weight_class, self.weight_mask, self.weight_dice,self.weight_box,self.weight_center = costs
        self.freepoint_loss = freepoint_loss

        assert (
            self.weight_class != 0 or self.weight_mask != 0 or self.weight_dice != 0
        ), "all costs cant be 0"

        self.p_ratio = p_ratio

    @torch.no_grad()
    def forward(self, outputs, targets,coords):
        """Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(outputs, targets,coords)

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets,coords):
        """More memory-friendly matching"""
        bs, num_queries, num_classes = outputs["pred_logits"].shape

        indices = []

        # Iterate through batch size
        for b in range(bs):
            out_prob = outputs["pred_logits"][b].softmax(-1)
            tgt_ids = targets["classes"][b].type(torch.int64)

            cost_class = -out_prob[:, tgt_ids]

            out_mask = outputs["pred_masks"][b].permute(1, 0)  # [num_queries, num_pts]
            tgt_mask = targets["masks"][b].to(out_mask)
            n_pts_scan = tgt_mask.shape[1]

            # all masks share the same set of points for efficient matching!
            pt_idx = torch.randint(
                0, n_pts_scan, (int(self.p_ratio * n_pts_scan), 1)
            ).squeeze(1)

            # get gt labels
            tgt_mask = tgt_mask[:, pt_idx]
            out_mask = out_mask[:, pt_idx]

            with autocast(enabled=False):
                out_mask = out_mask.float()  # [num_q,num_pts]
                tgt_mask = tgt_mask.float()  # [n_ins,num_pts]
                cost_mask = batch_sigmoid_ce_cost_jit(out_mask, tgt_mask)
                cost_dice = batch_dice_cost_jit(out_mask, tgt_mask)
                
                if self.freepoint_loss : 
                    loss_box = box_loss(out_mask,tgt_mask,coords)
                    cost_box = loss_box['box_loss']
                    cost_center = loss_box['loss_center']

            # Final cost matrix
            if self.freepoint_loss : 
                C = (
                    self.weight_mask * cost_mask
                    + self.weight_class * cost_class
                    + self.weight_dice * cost_dice 
                    + self.weight_box * cost_box
                    + self.weight_center * cost_center
                )
            
            else : 
                C = (
                    self.weight_mask * cost_mask
                    + self.weight_class * cost_class
                    + self.weight_dice * cost_dice
                )
            C = C.reshape(num_queries, -1).cpu()
            indices.append(lsa(C))


        return [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]

def distance_calc(matrix1,matrix2):
    matrix1_expanded = matrix1.unsqueeze(1)  # Shape becomes 40x1x3
    matrix2_expanded = matrix2.unsqueeze(0)  # Shape becomes 1x13x3

    squared_diff = (matrix1_expanded - matrix2_expanded) ** 2
    squared_distances = squared_diff.sum(dim=2)

    distances = torch.sqrt(squared_distances)   
    return distances

def box_loss(point_logits, point_labels, coords):
    probabilities = F.softmax(point_logits, dim=0)
    max_indices = torch.argmax(probabilities, dim=0)
    
    # Initialize a binary matrix with zeros
    binary_matrix = torch.zeros_like(point_logits, dtype=torch.int).cuda()
    
    # Use advanced indexing to set the corresponding positions to 1
    binary_matrix[max_indices, torch.arange(point_logits.shape[1])] = 1
    
    center_preds = torch.zeros((binary_matrix.shape[0],3)).cuda()
    center_gts = torch.zeros((point_labels.shape[0],3)).cuda()
    
    center_preds_mins = torch.zeros((binary_matrix.shape[0],3)).cuda()
    center_preds_maxs = torch.zeros((binary_matrix.shape[0],3)).cuda()

    center_gts_mins = torch.zeros((point_labels.shape[0],3)).cuda()
    center_gts_maxs = torch.zeros((point_labels.shape[0],3)).cuda()
    
    for d in range(binary_matrix.shape[0]): 
        cur_idcs_pred = torch.where(binary_matrix[d,:] == 1)[0]
        if cur_idcs_pred.shape[0] != 0: 
            center_preds[d] = coords[cur_idcs_pred].sum(0) / cur_idcs_pred.shape[0]
            center_preds_mins[d] = coords[cur_idcs_pred].min(0)[0]
            center_preds_maxs[d] = coords[cur_idcs_pred].max(0)[0]
    
    for j in range(point_labels.shape[0]):
        cur_idcs_tgt = torch.where(point_labels[j,:] == 1)[0]
        center_gts[j] = coords[cur_idcs_tgt].sum(0) / cur_idcs_tgt.shape[0]
        center_gts_mins[j] = coords[cur_idcs_tgt].min(0)[0]
        center_gts_maxs[j] = coords[cur_idcs_tgt].max(0)[0]
        
    center_dists = distance_calc(center_preds,center_gts)
    box_min_dists = distance_calc(center_preds_mins,center_gts_mins)
    box_max_dists = distance_calc(center_preds_maxs,center_gts_maxs)
    
    
    
    
    return {'box_loss':box_min_dists + box_max_dists,"loss_center":center_dists}
        
    
    

def batch_dice_cost(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss

def batch_iou(inputs,targets):
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = torch.einsum("nc,mc->nm", inputs, targets)
    denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
    iou = numerator/denominator
    return iou
    

#batch_dice_cost_jit = torch.jit.script(batch_dice_cost)  # type: torch.jit.ScriptModule
batch_dice_cost_jit = batch_dice_cost  # type: torch.jit.ScriptModule


def batch_sigmoid_ce_cost(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
        "nc,mc->nm", neg, (1 - targets)
    )

    return loss / hw


batch_sigmoid_ce_cost_jit = torch.jit.script(
    batch_sigmoid_ce_cost
)  # type: torch.jit.ScriptModule

