"""
SfM preprocessing module for OnePose ROS demo.

This module provides functionality to run SfM reconstruction for object point cloud
generation, similar to the sfm() function in run.py.
"""

import os
import glob
import json
from pathlib import Path
import os.path as osp
from loguru import logger

# Import SfM core functions
from onnx_demo.sfm import extract_features, match_features, generate_empty, triangulation, pairs_from_poses
from onnx_demo.sfm.postprocess import filter_points, feature_process, filter_tkl


def run_sfm(data_dir, outputs_dir, detection="superpoint", matching="superglue",
            down_ratio=5, covis_num=10, rotation_thresh=50, max_num_kp3d=2500,
            redo=True):
    """
    Run SfM reconstruction for object point cloud generation.
    
    Parameters
    ----------
    data_dir : str
        Data directory containing object sequences (e.g., "data/demo/obj_name obj_name-annotate")
    outputs_dir : str
        Output directory for SfM model (e.g., "data/demo/obj_name/sfm_model")
    detection : str
        Detection network name (default: "superpoint")
    matching : str
        Matching network name (default: "superglue")
    down_ratio : int
        Image downsampling ratio (default: 5)
    covis_num : int
        Number of covisible images for matching (default: 10)
    rotation_thresh : float
        Rotation threshold for covisibility (default: 50)
    max_num_kp3d : int
        Maximum number of 3D keypoints (default: 2500)
    redo : bool
        Whether to redo SfM if outputs already exist (default: True)
    
    Returns
    -------
    bool
        True if SfM succeeded, False otherwise
    """
    # Parse data directories
    data_dirs = [data_dir] if isinstance(data_dir, str) else data_dir
    
    for data_dir_item in data_dirs:
        logger.info(f"Processing {data_dir_item}.")
        root_dir, sub_dirs = data_dir_item.split(' ')[0], data_dir_item.split(' ')[1:]

        # Parse image directory and downsample images:
        img_lists = []
        for sub_dir in sub_dirs:
            seq_dir = osp.join(root_dir, sub_dir)
            img_lists += glob.glob(str(Path(seq_dir)) + '/color/*.png', recursive=True)

        down_img_lists = []
        for img_file in img_lists:
            index = int(img_file.split('/')[-1].split('.')[0])
            if index % down_ratio == 0:
                down_img_lists.append(img_file)  
        img_lists = down_img_lists

        if len(img_lists) == 0:
            logger.info(f"No png image in {root_dir}")
            continue
        
        obj_name = root_dir.split('/')[-1]
        outputs_dir_root = outputs_dir.format(obj_name)

        # Begin SfM and postprocess:
        success = sfm_core(img_lists, outputs_dir_root, detection, matching, 
                          covis_num, rotation_thresh, redo)
        if not success:
            return False
        
        success = postprocess(img_lists, root_dir, outputs_dir_root, detection, 
                             matching, max_num_kp3d)
        if not success:
            return False
    
    return True


def sfm_core(img_lists, outputs_dir_root, detection="superpoint", 
             matching="superglue", covis_num=10, rotation_thresh=50, redo=True):
    """
    Sparse reconstruction: extract features, match features, triangulation.
    """
    # Construct output directory structure:
    outputs_dir = osp.join(outputs_dir_root, 'outputs' + '_' + detection + '_' + matching)
    feature_out = osp.join(outputs_dir, f'feats-{detection}.h5')
    covis_pairs_out = osp.join(outputs_dir, f'pairs-covis{covis_num}.txt')
    matches_out = osp.join(outputs_dir, f'matches-{matching}.h5')
    empty_dir = osp.join(outputs_dir, 'sfm_empty')
    deep_sfm_dir = osp.join(outputs_dir, 'sfm_ws')
    
    if redo:
        import shutil
        if osp.exists(outputs_dir):
            shutil.rmtree(outputs_dir)
        Path(outputs_dir).mkdir(exist_ok=True, parents=True)

        # Extract image features, construct image pairs and then match:
        # Note: We need to create a minimal config for extract_features
        class SimpleConfig:
            def __init__(self):
                self.network = type('Network', (), {
                    'detection': detection,
                    'matching': matching
                })()
                self.sfm = type('Sfm', (), {
                    'covis_num': covis_num,
                    'rotation_thresh': rotation_thresh
                })()
        
        cfg = SimpleConfig()
        extract_features.main(img_lists, feature_out, cfg)
        pairs_from_poses.covis_from_pose(img_lists, covis_pairs_out, covis_num, 
                                         max_rotation=rotation_thresh)
        match_features.main(cfg, feature_out, covis_pairs_out, matches_out, vis_match=False)

        # Reconstruct 3D point cloud with known image poses:
        generate_empty.generate_model(img_lists, empty_dir)
        triangulation.main(deep_sfm_dir, empty_dir, outputs_dir, covis_pairs_out, 
                          feature_out, matches_out, image_dir=None)
    
    return True


def postprocess(img_lists, root_dir, outputs_dir_root, detection="superpoint",
                matching="superglue", max_num_kp3d=2500):
    """
    Filter points and average feature.
    """
    bbox_path = osp.join(root_dir, "box3d_corners.txt")
    # Construct output directory structure:
    outputs_dir = osp.join(outputs_dir_root, 'outputs' + '_' + detection + '_' + matching)
    feature_out = osp.join(outputs_dir, f'feats-{detection}.h5')
    deep_sfm_dir = osp.join(outputs_dir, 'sfm_ws')
    model_path = osp.join(deep_sfm_dir, 'model')

    if not osp.exists(model_path):
        logger.error(f"Model path not found: {model_path}")
        return False

    # Select feature track length to limit the number of 3D points below the 'max_num_kp3d' threshold:
    track_length, points_count_list = filter_tkl.get_tkl(model_path, thres=max_num_kp3d, show=False) 
    filter_tkl.vis_tkl_filtered_pcds(model_path, points_count_list, track_length, outputs_dir) # For visualization only

    # Leverage the selected feature track length threshold and 3D BBox to filter 3D points:
    xyzs, points_idxs = filter_points.filter_3d(model_path, track_length, bbox_path)
    # Merge 3d points by distance between points
    merge_xyzs, merge_idxs = filter_points.merge(xyzs, points_idxs, dist_threshold=1e-3) 

    # Save features of the filtered point cloud:
    class SimpleConfig:
        def __init__(self):
            self.network = type('Network', (), {
                'detection': detection,
                'matching': matching
            })()
    
    cfg = SimpleConfig()
    feature_process.get_kpt_ann(cfg, img_lists, feature_out, outputs_dir, merge_idxs, merge_xyzs)
    
    return True


def check_sfm_model_exists(sfm_model_dir, detection="superpoint", matching="superglue"):
    """
    Check if SfM model already exists with required files.
    
    Returns
    -------
    bool
        True if all required SfM output files exist
    """
    anno_dir = osp.join(sfm_model_dir, f"outputs_{detection}_{matching}", "anno")
    
    required_files = [
        osp.join(anno_dir, "anno_3d_average.npz"),
        osp.join(anno_dir, "anno_3d_collect.npz"),
        osp.join(anno_dir, "idxs.npy"),
    ]
    
    return all(osp.exists(f) for f in required_files)


if __name__ == "__main__":
    # For testing
    import sys
    if len(sys.argv) < 3:
        print("Usage: python sfm_preprocess.py <data_dir> <outputs_dir>")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    outputs_dir = sys.argv[2]
    
    success = run_sfm(data_dir, outputs_dir)
    if success:
        print("SfM preprocessing completed successfully")
    else:
        print("SfM preprocessing failed")
        sys.exit(1)