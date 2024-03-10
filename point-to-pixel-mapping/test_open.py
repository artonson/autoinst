import zlib


file = '/media/cedric/Datasets2/semantic_kitti/tarl_features/00/001479.bin'
file = '/media/cedric/Datasets2/semantic_kitti/tarl_features/02/002185.bin' 
print('file',file) 
with open(file, 'rb') as f_in:
        compressed_data = f_in.read()
try : 
    decompressed_data = zlib.decompress(compressed_data)
except : 
    decompressed_data = zlib.decompress(compressed_data)
#decompressed_data = zlib.decompress(compressed_data)
print("file opened") 


