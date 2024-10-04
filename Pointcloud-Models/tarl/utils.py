import MinkowskiEngine as ME
import torch

latent_features = {
    'SparseResNet14': 512,
    'SparseResNet18': 1024,
    'SparseResNet34': 2048,
    'SparseResNet50': 2048,
    'SparseResNet101': 2048,
    'MinkUNet': 96,
    'MinkUNetSMLP': 96,
    'MinkUNet14': 96,
    'MinkUNet18': 1024,
    'MinkUNet34': 2048,
    'MinkUNet50': 2048,
    'MinkUNet101': 2048,
}


def set_deterministic():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True


def array_to_sequence(batch_data):
    return [row for row in batch_data]


def array_to_torch_sequence(batch_data):
    return [torch.from_numpy(row).float() for row in batch_data]


def numpy_to_sparse_tensor(p_coord, p_feats):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(
        array_to_sequence(p_coord), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(
        array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]

    return ME.SparseTensor(
        features=p_feats,
        coordinates=p_coord,
        device=device,
    )
