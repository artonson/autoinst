import os
import numpy as np
import cv2
from tqdm import tqdm 
import random 
from scipy.spatial import cKDTree
random.seed(42)


images = ['image_3','image_2']
#images = ["CAM_FRONT_LEFT","CAM_FRONT_RIGHT"]

Overseg = True ##set to false to generate smaller masks
VIS = False 
out_small = False ##if True small masks are saved, else medium masks

def generate_unique_colors(n):
	colors = set()
	while len(colors) < n:
		colors.add((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
	return colors

def find_nearest_non_masked_pixel(tree,hole_pixel,non_hole_pixels):
    distance, index = tree.query(hole_pixel)
    return non_hole_pixels[index]



def calculate_overlap(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection)/np.sum(mask1) 

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection)/np.sum(union)
    
def calculate_overlap_large(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection)/np.sum(mask2) 

def generate_medium_masks(sam_masks,thresh_small=0.9,thresh_large=0.05,iou_thresh=0.9):
	masks_set = set()
	out_masks = []
	for i, mask_i in enumerate(sam_masks):
	    if i in masks_set:
	        continue  # Skip if this mask is already processed
	
	    idcs = np.where(mask_i['segmentation'] == 1)
	
	    ignore_mask = False
	    for j, mask_j in enumerate(sam_masks):
	        if i != j and mask_i['area'] < mask_j['area']:
	            overlap_small = calculate_overlap(mask_i['segmentation'], mask_j['segmentation'])
	            overlap_large = calculate_overlap_large(mask_i['segmentation'], mask_j['segmentation'])
	            iou = calculate_iou(mask_i['segmentation'], mask_j['segmentation'])
	
	            if (overlap_small > thresh_small and overlap_large < thresh_large) or (overlap_small > iou_thresh and np.abs(mask_i['area']/mask_j['area']) > 0.6):
	                ignore_mask = True
	                masks_set.add(j)
	                break
	
	    if not ignore_mask:
	        masks_set.add(i)
	        out_masks.append(mask_i)
	return out_masks

def generate_small_masks(sam_masks):
	filtered_masks = []
	cnt = 0
	for i, mask_i in enumerate(sam_masks):
	    ignore_mask = False
	    for j, mask_j in enumerate(sam_masks):
	        if i != j and mask_i['area'] < mask_j['area']:
	                overlap = calculate_overlap(mask_i['segmentation'], mask_j['segmentation'])
	                if overlap > 0.5:  # Define YOUR_OVERLAP_THRESHOLD
	                        ignore_mask = True
	                        break
	    if not ignore_mask:
	        try : 
	                mask_i['color'] = unique_colors[cnt]
	        except : 
	                breakpoint()
	        cnt += 1
	        filtered_masks.append(mask_i)
	return filtered_masks

unique_colors = list(generate_unique_colors(200)) 

sequences = [
	'00',
	]

for seq in sequences : 
	print("seq",seq)
	for image_name in images : 
		print("image name",image_name)
		
		directory_sam_feats = '/media/cedric/Datasets2/semantic_kitti/outputs/' + seq + '/' + image_name + '/'
		
		out_path = '/media/cedric/Datasets2/semantic_kitti/sam_pred_underseg/' + seq + '/' + image_name + '/'
		raw_imgs_dir = '/media/cedric/Datasets2/semantic_kitti/sequences/' + seq + '/' + image_name + '/'
		
		imgs = os.listdir(raw_imgs_dir)
		imgs = [img for img in imgs if img.endswith('.png')]
		
		if os.path.exists(out_path) == False : 
			os.makedirs(out_path)
	
		for img in tqdm(imgs) : 
			full_img_path = raw_imgs_dir + img
			img_name = img.split('.')[0]
			sam_masks = np.load(directory_sam_feats + img_name + '.npz',allow_pickle=True)['masks']
			img = cv2.imread(full_img_path)
			orig_img = img.copy()
			new_img = np.zeros_like(img) 
			new_img_medium = np.zeros_like(img)
			standard_img = np.zeros_like(img)
			
			for cnt,mask in enumerate(sam_masks): 
				box_size = np.sum(mask['segmentation'])
				mask['area'] = box_size
				mask['color'] = unique_colors[cnt]
			
			#filtered_masks = generate_small_masks(sam_masks)	
			filtered_masks_medium = generate_medium_masks(sam_masks)	
			
			orig_masks = sorted(sam_masks, key=lambda x: x['area'], reverse=True)
			#sorted_masks = sorted(filtered_masks, key=lambda x: x['area'], reverse=True)
			sorted_masks_medium = sorted(filtered_masks_medium, key=lambda x: x['area'], reverse=True)
			
			#sorted_masks_naive = sorted(sam_masks, key=lambda x: x['area'], reverse=Overseg)
			
			#num_masks = len(sorted_masks) 
			
			mask_medium_cols = []
			for idx,mask in enumerate(sorted_masks_medium):
				idcs = np.where(mask['segmentation'] == 1)
				new_img_medium[idcs] = mask['color']
				mask_medium_cols.append(mask['color'])
				#mask['color'] = unique_colors[idx]
			
			#for idx,mask in enumerate(orig_masks):
			#	idcs = np.where(mask['segmentation'] == 1)
			#	standard_img[idcs] = mask['color']
			
			
			#for idx,mask in enumerate(sorted_masks):
			#	idcs = np.where(mask['segmentation'] == 1)
			#	new_img[idcs] = mask['color']
			
			##apply closing operation to holes 
			
			gray_image = cv2.cvtColor(new_img_medium, cv2.COLOR_BGR2GRAY)
			_, binary_mask = cv2.threshold(gray_image, 10, 255, cv2.THRESH_BINARY)
			
			binary_mask = cv2.bitwise_not(binary_mask)
			num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
	
			# Filter out large components
			max_area = 500  # Define the maximum area of a hole
			for i in range(1, num_labels):
			    if stats[i, cv2.CC_STAT_AREA] > max_area:
			        labels[labels == i] = 0
			
			# Create a mask of the small holes
			small_holes_mask = np.uint8(labels != 0) * 255
			
			# Find coordinates of hole pixels and non-hole pixels
			hole_pixels = np.column_stack(np.where(small_holes_mask > 0))
			non_hole_pixels = np.column_stack(np.where(small_holes_mask == 0))
			
			# Create a copy of the image to modify
			morphed_img = new_img_medium.copy()
			
			tree = cKDTree(non_hole_pixels)
			for i,hole_pixel in enumerate(hole_pixels):
			    nearest_pixel = find_nearest_non_masked_pixel(tree,hole_pixel,non_hole_pixels)
			    ## store back to the old format
			    mask_id_new = mask_medium_cols.index(tuple(new_img_medium[tuple(nearest_pixel)]))
			    filtered_masks_medium[mask_id_new]['segmentation'][tuple(hole_pixel)] = True 
			    morphed_img[tuple(hole_pixel)] = new_img_medium[tuple(nearest_pixel)]
			
			if VIS == True : 
				img_vis = cv2.hconcat([standard_img,new_img_medium,morphed_img])
				img_vis = cv2.resize(img_vis, (int(img_vis.shape[1] * 0.7),int(img_vis.shape[0] * 0.7) ), interpolation=cv2.INTER_AREA)
				#cv2.imwrite('out_imgs/standard_img_' + img_name + '.png',standard_img)
				#cv2.imwrite('out_imgs/medium_mask_img_' + img_name + '.png',new_img_medium)
				#cv2.imwrite('out_imgs/morped_img_' + img_name + '.png',morphed_img)
				#cv2.imwrite('out_imgs/original' + img_name + '.png',orig_img)
				
				cv2.imshow('img',img_vis)
				cv2.waitKey(0)
			out_name = out_path + img_name + '.png'
			out_save = out_path + img_name + '.npz'
			if out_small == True:
				np.savez_compressed(out_save, masks = filtered_masks)
			else : 
				np.savez_compressed(out_save, masks = filtered_masks_medium)
