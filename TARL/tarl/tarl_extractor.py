import numpy as np
import torch
import MinkowskiEngine as ME
import os
from tqdm import tqdm
from minkunet import MinkUNet
import zlib
import gc

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
        return [ row for row in batch_data ]

def array_to_torch_sequence(batch_data):
    return [ torch.from_numpy(row).float() for row in batch_data ]

def numpy_to_sparse_tensor(p_coord, p_feats):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_coord = ME.utils.batched_coordinates(array_to_sequence(p_coord), dtype=torch.float32)
    p_feats = ME.utils.batched_coordinates(array_to_torch_sequence(p_feats), dtype=torch.float32)[:, 1:]

    return ME.SparseTensor(
                features=p_feats,
                coordinates=p_coord,
                device=device,
            )

class TARLFOLDER():
    def __init__(self,args, vis=False):

        self.vis = vis
        self.out_dir = args.output_path
        self.input_dir = args.input_path
        self.dataset = args.dataset 
        self.label_clusters = False 
        if os.path.exists(self.out_dir) == False:
            #    shutil.rmtree(self.out_dir)
            os.makedirs(self.out_dir)
        
        self.sparse_resolution = 0.05  ##according to standard TARL settings
        dtype = torch.cuda.FloatTensor
        self.use_intensity = True
        
        self.minkunet =  MinkUNet(
        in_channels=4 if self.use_intensity else 3,
        out_channels=latent_features['MinkUNet'],
        )#.type(dtype)

        self.minkunet.cuda()
        self.minkunet.eval()
        set_deterministic()
        
        model_filename = f'tarl.pt'
        checkpoint = torch.load(model_filename)
        self.minkunet.load_state_dict(checkpoint['model'])
        self.max_len = 1091 
        self.device = torch.device("cuda")
        torch.cuda.set_device(0)
    
    #@profile
    def extract_instance(self, pts,name):

        points,point_feats, mapping_idcs = self.point_set_to_coord_feats(pts, deterministic=True)
        points = np.asarray(points)[None,:]
        point_feats = np.asarray(point_feats)[None,:]
        x = numpy_to_sparse_tensor(points, point_feats) 
        h = self.minkunet(x)
        minkunet_feats = h.features.cpu() #96 dim features of minkunet 
        
        
        point_feats_minkunet = torch.zeros((pts.shape[0],96)) ###create original size of point cloud features 
        extracted_features = torch.zeros((pts.shape[0],1)) ##to save indices accordingly 
        extracted_features[mapping_idcs] = 1
        non_mapped_idcs = torch.nonzero(extracted_features == 0).squeeze().reshape(-1,2)[:,0]

        
        
        '''
        as minkunet takes in quantized points, we assign the features of the closest point to the 
        points that are not mapped 
        '''
        point_feats_minkunet[mapping_idcs] = minkunet_feats.detach()
        
        distances = torch.cdist(torch.from_numpy(pts[non_mapped_idcs,:3]).cuda(),torch.from_numpy(pts[mapping_idcs,:3]).cuda()) #speedup by moving to GPU
        
        closest_indices = torch.argmin(distances, dim=1).cpu()
        
        point_feats_minkunet[non_mapped_idcs] = point_feats_minkunet[mapping_idcs][closest_indices]
    
        
        ## loop over clusters and assign the mean features to each cluster

            
        #np.save(self.out_dir + output_name, np.concatenate((pts,mean_feats.detach().numpy(),point_feats_minkunet.detach().numpy()),axis=1))
        data_store = point_feats_minkunet.detach().numpy()
        #with gzip.open(self.out_dir + output_name + '.gz', 'wb') as f_out:
        #        np.save(f_out, data_store)
        output_name = name.split('.')[0] + '.bin'
        with open(self.out_dir + output_name, 'wb') as f_out:
                f_out.write(zlib.compress(data_store.tobytes())) 
        
        del distances , x, h, points, non_mapped_idcs
        torch.cuda.empty_cache() 
        del closest_indices, extracted_features,pts 
        del data_store
        del point_feats_minkunet
        del point_feats, mapping_idcs
        gc.collect()

        
    
    def point_set_to_coord_feats(self,point_set, deterministic=False):
        p_feats = point_set.copy()
        p_coord = np.round(point_set[:, :3] / self.sparse_resolution)
        p_coord -= p_coord.min(0, keepdims=1)

        _, mapping = ME.utils.sparse_quantize(coordinates=p_coord, return_index=True)

        return p_coord[mapping], p_feats[mapping],mapping
        
    def run_on_folder(self): 
        all_lidars = os.listdir(self.input_dir)
        all_lidars = sorted(all_lidars, key=lambda x: int(x.split('.')[0]))

        for lidar_name in tqdm(all_lidars):
            if self.dataset == 'nuscenes':
            	points = np.fromfile(os.path.join(self.input_dir, lidar_name), dtype=np.float32).reshape(-1,5)[:,:4]
            elif self.dataset == 'kitti': 
                points = np.fromfile(os.path.join(self.input_dir, lidar_name), dtype=np.float32).reshape(-1,4)
            else : 
                raise "Dataset" +  self.dataset + "does not exist"
            self.extract_instance(points,lidar_name)
        
